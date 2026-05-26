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


class AdapterLayer(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        d_in = linear.in_features
        d_out = linear.out_features

        self.down = nn.Parameter(torch.empty(rank, d_in))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.up = nn.Parameter(torch.zeros(d_out, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.linear(x) + (x @ self.down.T @ self.up.T) * self.scaling

    def task_vector(self):
        return torch.cat([self.down.data.flatten(), self.up.data.flatten()])

    def load_task_vector(self, flat):
        d_rank, d_in = self.down.shape
        d_out = self.up.shape[0]
        n_down = d_rank * d_in
        self.down.data = flat[:n_down].reshape(d_rank, d_in)
        self.up.data = flat[n_down:].reshape(d_out, d_rank)


class LoRAViT(nn.Module):
    def __init__(self, n_classes, rank=16, alpha=32, adapter_targets=None, cosine_scale=20.0):
        super().__init__()
        self.vit = timm.create_model("deit_small_patch16_224", pretrained=False)
        self.vit = load_pretrain(self.vit)
        self.feature_dim = 384
        self.rank = rank
        self.alpha = alpha
        self.cosine_scale = cosine_scale

        for param in self.vit.parameters():
            param.requires_grad = False

        if adapter_targets is None:
            adapter_targets = ["qkv", "proj"]

        def _get_linear(block, name):
            if name == "qkv": return block.attn.qkv
            if name == "proj": return block.attn.proj
            if name == "fc1": return block.mlp.fc1
            if name == "fc2": return block.mlp.fc2
            return None

        self._adapter_info = []
        self._original_linears = {}
        for block_idx, block in enumerate(self.vit.blocks):
            for name in adapter_targets:
                linear = _get_linear(block, name)
                if linear is None:
                    continue
                self._adapter_info.append((block_idx, name))
                self._original_linears[(block_idx, name)] = linear

        self.vit.head = nn.Identity()
        self.head = CosineLinear(self.feature_dim, n_classes, scale=cosine_scale)

        self.task_adapters = nn.ModuleList()
        self.universal_adapter = None
        self._active_adapter_idx = -1

    def create_task_adapter(self):
        adapter = nn.ModuleList()
        for block_idx, name in self._adapter_info:
            orig = self._original_linears[(block_idx, name)]
            adapter.append(AdapterLayer(orig, self.rank, alpha=self.alpha))
        adapter = adapter.to(self.vit.pos_embed.device)
        self.task_adapters.append(nn.ModuleList(adapter))
        idx = len(self.task_adapters) - 1
        return idx

    def activate_adapter(self, idx):
        self._active_adapter_idx = idx
        if idx < 0:
            self._remove_adapters()
            return
        layers = self.task_adapters[idx]
        for layer, (block_idx, name) in zip(layers, self._adapter_info):
            block = self.vit.blocks[block_idx]
            if name == "qkv":
                block.attn.qkv = layer
            elif name == "proj":
                block.attn.proj = layer
            elif name == "fc1":
                block.mlp.fc1 = layer
            elif name == "fc2":
                block.mlp.fc2 = layer

    def activate_universal(self):
        if self.universal_adapter is None:
            return
        self._active_adapter_idx = -2
        for layer, (block_idx, name) in zip(self.universal_adapter, self._adapter_info):
            block = self.vit.blocks[block_idx]
            if name == "qkv":
                block.attn.qkv = layer
            elif name == "proj":
                block.attn.proj = layer
            elif name == "fc1":
                block.mlp.fc1 = layer
            elif name == "fc2":
                block.mlp.fc2 = layer

    def _remove_adapters(self):
        for block_idx, name in self._adapter_info:
            block = self.vit.blocks[block_idx]
            orig = self._original_linears[(block_idx, name)]
            if name == "qkv":
                block.attn.qkv = orig
            elif name == "proj":
                block.attn.proj = orig
            elif name == "fc1":
                block.mlp.fc1 = orig
            elif name == "fc2":
                block.mlp.fc2 = orig

    def _run_with_adapter(self, x, adapter_layers):
        for layer, (block_idx, name) in zip(adapter_layers, self._adapter_info):
            block = self.vit.blocks[block_idx]
            if name == "qkv":
                block.attn.qkv = layer
            elif name == "proj":
                block.attn.proj = layer
            elif name == "fc1":
                block.mlp.fc1 = layer
            elif name == "fc2":
                block.mlp.fc2 = layer
        out = self.vit(x)
        if self._active_adapter_idx >= 0:
            self.activate_adapter(self._active_adapter_idx)
        elif self._active_adapter_idx == -2:
            self.activate_universal()
        else:
            self._remove_adapters()
        return out

    @torch.no_grad()
    def fuse_adapters(self):
        t = len(self.task_adapters)
        if t == 0:
            return

        vectors = []
        for adapter in self.task_adapters:
            parts = []
            for layer in adapter:
                parts.append(layer.task_vector())
            vectors.append(torch.cat(parts))

        stacked = torch.stack(vectors)
        sign_sum = torch.sign(stacked.sum(dim=0))

        pos_max = stacked.clamp(min=0).max(dim=0).values
        neg_max = (-stacked).clamp(min=0).max(dim=0).values
        zero = torch.zeros_like(sign_sum)
        mag = torch.where(sign_sum > 0, pos_max, torch.where(sign_sum < 0, neg_max, zero))

        uni_vec = mag * sign_sum

        offset = 0
        uni_layers = nn.ModuleList()
        for block_idx, name in self._adapter_info:
            orig = self._original_linears[(block_idx, name)]
            layer = AdapterLayer(orig, self.rank, alpha=self.alpha)
            tv_size = layer.task_vector().shape[0]
            layer.load_task_vector(uni_vec[offset:offset + tv_size])
            layer.down.requires_grad_(False)
            layer.up.requires_grad_(False)
            uni_layers.append(layer)
            offset += tv_size

        self.universal_adapter = uni_layers.to(self.vit.pos_embed.device)

    def features_with_task(self, x, task_idx):
        adapter = self.task_adapters[task_idx]
        return self._run_with_adapter(x, adapter)

    def features_universal(self, x):
        if self.universal_adapter is None:
            return self.vit(x)
        return self._run_with_adapter(x, self.universal_adapter)

    def features(self, x):
        return self.vit(x)

    def forward(self, x):
        return self.head(self.features(x))

    def current_adapter_params(self):
        if self._active_adapter_idx < 0:
            return []
        adapter = self.task_adapters[self._active_adapter_idx]
        params = []
        for layer in adapter:
            params.append(layer.down)
            params.append(layer.up)
        return params

    def get_up_weights(self, task_idx):
        adapter = self.task_adapters[task_idx]
        ups = []
        for layer in adapter:
            ups.append(layer.up)
        return ups
