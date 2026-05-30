import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from utils.train_utils import load_pretrain


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, scale=20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.scale = scale

    def forward(self, x):
        return F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1)) * self.scale


class InfLoRALayer(nn.Module):
    """Frozen linear with a growing list of InfLoRA branches.

    Forward: y = W x + sum_j (alpha/r) * (x @ B_j.T) @ A_j.T

    B_j is pre-designed before learning task j by SVD on input features projected
    onto the orthogonal complement of the past-task input subspace (M_basis).
    A_j is zero-initialized and only trainable while task j is active.
    """

    def __init__(self, linear, rank, alpha, calib_cap=2048, calib_per_batch=16):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        self.d_in = linear.in_features
        self.d_out = linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A_list = nn.ParameterList()
        self.B_list = nn.ParameterList()

        self.register_buffer("M_basis", torch.zeros(self.d_in, 0))

        self._calib_buf = []
        self._collecting = False
        self.calib_cap = calib_cap
        self.calib_per_batch = calib_per_batch

    def forward(self, x):
        out = self.linear(x)
        for A, B in zip(self.A_list, self.B_list):
            out = out + self.scaling * (x @ B.T) @ A.T

        if self._collecting:
            with torch.no_grad():
                flat = x.detach().reshape(-1, self.d_in)
                n_keep = min(self.calib_per_batch, flat.shape[0])
                if flat.shape[0] > n_keep:
                    idx = torch.randperm(flat.shape[0], device=flat.device)[:n_keep]
                    flat = flat[idx]
                self._calib_buf.append(flat)
                total = sum(c.shape[0] for c in self._calib_buf)
                # Reservoir-like cap: when over capacity, drop oldest chunks
                # until we're back under the cap.
                while total > self.calib_cap and len(self._calib_buf) > 1:
                    total -= self._calib_buf.pop(0).shape[0]

        return out

    def start_collecting(self):
        self._calib_buf = []
        self._collecting = True

    def stop_collecting(self):
        self._collecting = False

    def _drain_calib(self, max_samples=None):
        if len(self._calib_buf) == 0:
            return None
        H = torch.cat(self._calib_buf, dim=0)
        self._calib_buf = []
        if max_samples is not None and H.shape[0] > max_samples:
            idx = torch.randperm(H.shape[0], device=H.device)[:max_samples]
            H = H[idx]
        return H

    @torch.no_grad()
    def add_task_branch(self, max_samples=2048):
        """Design B_t from current _calib_buf and add a new branch.

        B_t rows lie in span(N_t ∩ M_t^⊥): the part of the new task's input
        space orthogonal to past tasks' input space.
        """
        H = self._drain_calib(max_samples=max_samples)
        device = self.linear.weight.device
        if H is None or H.shape[0] == 0:
            B_t = F.normalize(torch.randn(self.rank, self.d_in, device=device), dim=1)
        else:
            if self.M_basis.shape[1] > 0:
                M = self.M_basis
                H = H - (H @ M) @ M.T
            Ht = H.T  # [d_in, n]
            try:
                U, _, _ = torch.linalg.svd(Ht, full_matrices=False)
            except RuntimeError:
                q = min(self.rank * 4, Ht.shape[1])
                U, _, _ = torch.svd_lowrank(Ht, q=q)
            r = min(self.rank, U.shape[1])
            B_t = U[:, :r].T.contiguous()
            if r < self.rank:
                pad = F.normalize(
                    torch.randn(self.rank - r, self.d_in, device=device), dim=1
                )
                B_t = torch.cat([B_t, pad], dim=0)

        # Freeze previously-active A (if any).
        for A in self.A_list:
            A.requires_grad_(False)
        # B is frozen by design.
        B_param = nn.Parameter(B_t.detach().to(device), requires_grad=False)
        A_param = nn.Parameter(torch.zeros(self.d_out, self.rank, device=device))
        self.A_list.append(A_param)
        self.B_list.append(B_param)

    @torch.no_grad()
    def update_M_basis(self, max_samples=2048, energy_threshold=0.99):
        """Extend M_basis to span past-task input subspace ∪ this task's
        residual (component outside current M_basis)."""
        H = self._drain_calib(max_samples=max_samples)
        if H is None or H.shape[0] == 0:
            return
        device = self.linear.weight.device

        if self.M_basis.shape[1] > 0:
            M = self.M_basis
            H_residual = H - (H @ M) @ M.T
            # Concatenate existing basis (as columns) with new candidate columns.
            C = torch.cat([M, H_residual.T], dim=1)  # [d_in, k]
        else:
            C = H.T

        try:
            U, S, _ = torch.linalg.svd(C, full_matrices=False)
        except RuntimeError:
            q = min(C.shape[1], C.shape[0])
            U, S, _ = torch.svd_lowrank(C, q=q)

        if S.numel() == 0:
            return
        energy = S * S
        cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp(min=1e-12)
        keep_mask = cumulative <= energy_threshold
        keep = int(keep_mask.sum().item()) + 1
        keep = min(keep, U.shape[1])
        self.M_basis = U[:, :keep].detach().to(device).contiguous()


class InfLoRAViT(nn.Module):
    """DeiT-S/16 with per-task InfLoRA branches at every (qkv, proj, fc1, fc2)
    projection. Pre-trained weights and old branches are frozen; only the
    current task's A_t plus the cosine classifier head are trainable.
    """

    def __init__(
        self,
        n_classes,
        rank=16,
        alpha=32,
        adapter_targets=None,
        cosine_scale=20.0,
        calib_cap=2048,
        calib_per_batch=16,
    ):
        super().__init__()
        self.vit = timm.create_model("deit_small_patch16_224", pretrained=False)
        self.vit = load_pretrain(self.vit)
        self.feature_dim = 384
        self.rank = rank
        self.alpha = alpha

        for param in self.vit.parameters():
            param.requires_grad = False

        if adapter_targets is None:
            adapter_targets = ["qkv", "proj", "fc1", "fc2"]

        def _get(block, name):
            return {
                "qkv": block.attn.qkv,
                "proj": block.attn.proj,
                "fc1": block.mlp.fc1,
                "fc2": block.mlp.fc2,
            }.get(name)

        def _set(block, name, layer):
            if name == "qkv":
                block.attn.qkv = layer
            elif name == "proj":
                block.attn.proj = layer
            elif name == "fc1":
                block.mlp.fc1 = layer
            elif name == "fc2":
                block.mlp.fc2 = layer

        self._adapter_info = []
        self.adapters = nn.ModuleList()
        for block_idx, block in enumerate(self.vit.blocks):
            for name in adapter_targets:
                linear = _get(block, name)
                if linear is None:
                    continue
                wrapped = InfLoRALayer(
                    linear, rank, alpha,
                    calib_cap=calib_cap, calib_per_batch=calib_per_batch,
                )
                _set(block, name, wrapped)
                self._adapter_info.append((block_idx, name))
                self.adapters.append(wrapped)

        self.vit.head = nn.Identity()
        self.head = CosineLinear(self.feature_dim, n_classes, scale=cosine_scale)
        self._current_task = -1

    def start_collecting(self):
        for layer in self.adapters:
            layer.start_collecting()

    def stop_collecting(self):
        for layer in self.adapters:
            layer.stop_collecting()

    @torch.no_grad()
    def design_new_branches(self, max_samples=2048):
        for layer in self.adapters:
            layer.add_task_branch(max_samples=max_samples)
        self._current_task += 1

    @torch.no_grad()
    def update_memory(self, max_samples=2048, energy_threshold=0.99):
        for layer in self.adapters:
            layer.update_M_basis(
                max_samples=max_samples, energy_threshold=energy_threshold
            )

    def current_task_params(self):
        params = list(self.head.parameters())
        for layer in self.adapters:
            if len(layer.A_list) > 0:
                A = layer.A_list[-1]
                if A.requires_grad:
                    params.append(A)
        return params

    def features(self, x):
        return self.vit(x)

    def forward(self, x):
        return self.head(self.features(x))
