import datetime
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from methods._trainer import _Trainer
from utils.online_sampler import OnlineTestSampler


class InfLoRA(_Trainer):
    """InfLoRA baseline for online GCIL (Si-Blurry).

    Per task t:
      1. Warmup phase (first `inflora_warmup` samples): collect input features
         H_t per linear layer; train only the head (so anytime-accuracy still
         improves while H_t accumulates).
      2. Pre-design B_t: project H_t onto past-task input subspace's orthogonal
         complement, SVD, take top-r left singular vectors of (H̃_t)^T.
      3. Add new branch (A_t=0, B_t frozen), rebuild optimizer with head + A_t.
      4. Stream-train A_t with cross-entropy.
      5. At task end: update DualGPM-style M_basis per layer to include the
         span of the new task's input features.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.warmup_samples = kwargs.get("inflora_warmup", 256)
        self.svd_max_samples = kwargs.get("inflora_svd_samples", 2048)
        self.m_energy = kwargs.get("inflora_m_energy", 0.99)
        self._cur_task_id = -1
        self._collected = 0
        self._scheduler_step_count = 0

    # ------------------------------------------------------------------ setup
    def setup_model(self):
        self.model = self.model.to(self.device)
        self.model_without_ddp = self.model
        # Only the head is trainable until the first branch is added.
        self.optimizer = self._build_optimizer(list(self.model.head.parameters()))
        self.scheduler = self._build_scheduler()
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters:\t{n_total}")

    def _build_optimizer(self, params):
        params = list(params)
        if self.opt_name == "adam":
            return torch.optim.Adam(params, lr=self.lr, weight_decay=0)
        if self.opt_name == "sgd":
            return torch.optim.SGD(
                params, lr=self.lr, momentum=0.9, nesterov=True, weight_decay=1e-4
            )
        return torch.optim.Adam(params, lr=self.lr)

    def _build_scheduler(self):
        if "cos" in self.sched_name:
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=1, T_mult=2
            )
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1)

    def _rebuild_optimizer(self):
        params = self.model.current_task_params()
        self.optimizer = self._build_optimizer(params)
        self.scheduler = self._build_scheduler()
        self._scheduler_step_count = 0
        n = sum(p.numel() for p in params)
        print(f"Trainable Parameters (head + A_t):\t{n}")

    # ----------------------------------------------------------- task hooks
    def online_before_task(self, task_id):
        self._cur_task_id = task_id
        self._collected = 0
        # Reset to head-only optimizer for the warmup window of this task.
        self.optimizer = self._build_optimizer(list(self.model.head.parameters()))
        self.scheduler = self._build_scheduler()
        self.model.start_collecting()
        print(
            f"[Task {task_id}] Warmup begins; collect {self.warmup_samples} samples "
            f"before designing B_t"
        )

    def online_after_task(self, task_id):
        # In case the task finishes before warmup completed (very small task),
        # still expand a branch so the model captures something for this task.
        if self.model._current_task < task_id:
            print(
                f"[Task {task_id}] Task ended before warmup threshold "
                f"({self._collected}/{self.warmup_samples}); designing B_t now"
            )
            self.model.design_new_branches(max_samples=self.svd_max_samples)
            self._rebuild_optimizer()
            self.model.start_collecting()

        # Use everything still in the calibration buffer to extend M_basis.
        self.model.update_memory(
            max_samples=self.svd_max_samples, energy_threshold=self.m_energy
        )
        self.model.stop_collecting()
        ks = [layer.M_basis.shape[1] for layer in self.model.adapters]
        print(
            f"[Task {task_id}] DualGPM M_basis updated; ranks per layer "
            f"min={min(ks)} max={max(ks)} mean={sum(ks) / len(ks):.1f}"
        )

    # ----------------------------------------------------------- train loop
    def train_learner(self):
        eval_dict = dict()
        self.model.eval()
        for images, labels, idx in self.train_dataloader:
            self.samples_cnt += images.size(0)
            result = self.online_step(images, labels, idx)
            self.report_training(self.samples_cnt, result["acc"])
            self._wandb.log(
                {
                    "train/loss_cls": result["loss_cls"],
                    "train/acc": result["acc"],
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "train/warmup": int(self.model._current_task < self._cur_task_id),
                },
                step=self.samples_cnt,
            )

            if self.samples_cnt > self.num_eval:
                with torch.no_grad():
                    test_sampler = OnlineTestSampler(
                        self.test_dataset, self.exposed_classes
                    )
                    test_dataloader = DataLoader(
                        self.test_dataset,
                        batch_size=self.batchsize * 8,
                        shuffle=False,
                        sampler=test_sampler,
                        num_workers=self.n_worker,
                    )
                    eval_dict = self.online_evaluate(test_dataloader)
                    self.eval_results["test_acc"].append(eval_dict["avg_acc"])
                    self.eval_results["avg_acc"].append(eval_dict["cls_acc"])
                    self.eval_results["data_cnt"].append(self.samples_cnt)
                    self.report_test(self.samples_cnt, eval_dict["avg_acc"])
                    n_eval_pts = len(self.eval_results["test_acc"])
                    a_avg_running = float(
                        sum(self.eval_results["test_acc"]) / max(1, n_eval_pts)
                    )
                    self._wandb.log(
                        {
                            "eval/test_acc": eval_dict["avg_acc"],
                            "eval/A_last": eval_dict["avg_acc"],
                            "eval/A_avg_running": a_avg_running,
                            "eval/task_id": self._cur_task_id,
                            "eval/num_classes": len(self.exposed_classes),
                            "eval/samples": self.samples_cnt,
                        },
                        step=self.samples_cnt,
                    )
                    self.num_eval += self.eval_period
            sys.stdout.flush()
        if len(eval_dict) != 0:
            self.report_test(self.samples_cnt, eval_dict["avg_acc"])

    def _maybe_promote(self):
        """If we've collected enough warmup samples, design B_t and switch
        from head-only training to head+A_t training. Idempotent."""
        if self.model._current_task >= self._cur_task_id:
            return
        if self._collected < self.warmup_samples:
            return
        print(
            f"[Task {self._cur_task_id}] Warmup done at {self._collected} samples; "
            f"designing B_t and adding new branch"
        )
        self.model.design_new_branches(max_samples=self.svd_max_samples)
        self._rebuild_optimizer()
        # Keep collecting throughout the rest of the task so M_basis update at
        # task end has a representative sample.
        self.model.start_collecting()

    def online_step(self, X, y, idx):
        self.add_new_class(y)
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())
        X = X.to(self.device)
        y = y.to(self.device)

        self._maybe_promote()
        warmup_phase = self.model._current_task < self._cur_task_id

        _acc, _iter, _loss = 0.0, 0, 0.0
        for _ in range(int(self.online_iter)):
            self.model.train()
            feat = self.model.features(X)
            logits = self.model.head(feat)
            loss = F.cross_entropy(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self._scheduler_step_count += 1
            if "cos" in self.sched_name:
                self.scheduler.step()

            with torch.no_grad():
                self.model.eval()
                feat_eval = self.model.features(X)
                logits_eval = self.model.head(feat_eval)
                pred = logits_eval.argmax(dim=1)
                acc = (pred == y).float().mean().item()

            _acc += acc
            _iter += 1
            _loss += loss.item()

        if warmup_phase:
            self._collected += X.size(0)

        return {"acc": _acc / _iter, "loss_cls": _loss / _iter}

    # ------------------------------------------------------------- evaluate
    def online_evaluate(self, test_loader):
        total_correct = 0.0
        total_num = 0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        self.model.eval()
        # Don't pollute the calibration buffer with test features.
        was_collecting = any(layer._collecting for layer in self.model.adapters)
        if was_collecting:
            self.model.stop_collecting()
        with torch.no_grad():
            for data in test_loader:
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())
                logits = self.model(x)
                _, pred = logits.topk(self.topk, 1, True, True)
                total_correct += torch.sum(pred == y.unsqueeze(1)).item()
                total_num += y.size(0)
                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred[:, 0])
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()
        if was_collecting:
            self.model.start_collecting()
        avg_acc = total_correct / total_num
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        return {"avg_acc": avg_acc, "cls_acc": cls_acc}

    # --------------------------------------------------------- reporting
    def report_training(self, sample_num, train_acc):
        elapsed = int(time.time() - self.start_time)
        eta = int(elapsed * (self.total_samples - sample_num) / max(sample_num, 1))
        print(
            f"Train | Sample # {sample_num} | train_acc {train_acc:.4f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            f"running_time {datetime.timedelta(seconds=elapsed)} | "
            f"ETA {datetime.timedelta(seconds=eta)}"
        )

    def report_test(self, sample_num, avg_acc):
        print(f"Test | Sample # {sample_num} | test_acc {avg_acc:.4f} | ")
