import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import time
import sys
from utils.online_sampler import OnlineTestSampler
from torch.utils.data import DataLoader
from methods._trainer import _Trainer


class FLAME(_Trainer):
    def __init__(self, **kwargs):
        super(FLAME, self).__init__(**kwargs)
        self.ca_lr = kwargs.get("ca_lr", 0.005)
        self.ca_epochs = kwargs.get("ca_epochs", 10)
        self.ca_samples = kwargs.get("ca_samples", 256)
        self.shrink_k = kwargs.get("shrink_k", 10.0)
        self._cur_task_id = -1
        self.cls_mean = {}
        self.cls_cov = {}
        self.uni_cls_mean = {}
        self.uni_cls_cov = {}
        self._prev_exposed = set()
        self._scheduler_step_count = 0

    def setup_model(self):
        self.model = self.model.to(self.device)
        self.model_without_ddp = self.model

        self.optimizer = self._build_optimizer(list(self.model.head.parameters()))
        self.scheduler = self._build_scheduler()

        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters:\t{n_total}")

    def _get_trainable_params(self):
        params = list(self.model.head.parameters())
        for p in self.model.current_adapter_params():
            params.append(p)
        return params

    def _build_optimizer(self, model_or_params):
        if isinstance(model_or_params, nn.Module):
            ps = list(model_or_params.parameters())
        else:
            ps = list(model_or_params)
        if self.opt_name == "adam":
            return torch.optim.Adam(ps, lr=self.lr, weight_decay=0)
        elif self.opt_name == "sgd":
            return torch.optim.SGD(ps, lr=self.lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
        return torch.optim.Adam(ps, lr=self.lr)

    def _build_scheduler(self):
        if "cos" in self.sched_name:
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=1, T_mult=2)
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1)

    def _rebuild_optimizer(self):
        self.optimizer = self._build_optimizer(self._get_trainable_params())
        self.scheduler = self._build_scheduler()
        self._scheduler_step_count = 0
        n_train = sum(p.numel() for p in self._get_trainable_params())
        print(f"Trainable Parameters (adapter+head):\t{n_train}")

    def online_before_task(self, task_id):
        self._cur_task_id = task_id
        self._task_proto_sums = {}
        self._task_proto_sum_sq = {}
        self._task_proto_counts = {}
        self._uni_proto_sums = {}
        self._uni_proto_sum_sq = {}
        self._uni_proto_counts = {}
        adapter_idx = self.model.create_task_adapter()
        self.model.activate_adapter(adapter_idx)
        self._rebuild_optimizer()
        print(f"[Task {task_id}] Created adapter {adapter_idx}")

    def online_after_task(self, task_id):
        self._compute_class_statistics(task_id)
        self._compute_universal_class_statistics(task_id)
        self._prev_exposed = set(self.exposed_classes)
        self.model.fuse_adapters()
        print(f"[Task {task_id}] Fused {len(self.model.task_adapters)} adapters")
        if task_id > 0:
            self._classifier_align()
        self.model.activate_universal()
        print(f"[Task {task_id}] Switched to universal adapter for evaluation")

    def _accumulate_into(self, sums, sum_sq, counts, features, labels):
        feat = features.detach()
        sq = feat * feat
        for c in labels.unique().tolist():
            mask = labels == c
            s = feat[mask].sum(dim=0)
            ss = sq[mask].sum(dim=0)
            n = int(mask.sum().item())
            if c not in sums:
                sums[c] = s.clone()
                sum_sq[c] = ss.clone()
                counts[c] = n
            else:
                sums[c] += s
                sum_sq[c] += ss
                counts[c] += n

    def _accumulate_features(self, features, labels):
        self._accumulate_into(self._task_proto_sums, self._task_proto_sum_sq, self._task_proto_counts, features, labels)

    def _accumulate_universal_features(self, features, labels):
        self._accumulate_into(self._uni_proto_sums, self._uni_proto_sum_sq, self._uni_proto_counts, features, labels)

    def _shrink_var(self, var, n):
        var = var.clamp(min=1e-4)
        alpha = min(1.0, self.shrink_k / max(n, 1))
        return (1 - alpha) * var + alpha * var.mean()

    @torch.no_grad()
    def _compute_class_statistics(self, task_id):
        for c in self._task_proto_sums:
            n = self._task_proto_counts[c]
            mean = self._task_proto_sums[c] / n
            var = (self._task_proto_sum_sq[c] / n) - (mean * mean)
            self.cls_mean[c] = mean
            self.cls_cov[c] = self._shrink_var(var, n)
        print(f"[Task {task_id}] Computed statistics for {len(self._task_proto_sums)} classes (total: {len(self.cls_mean)})")
        self._task_proto_sums = {}
        self._task_proto_sum_sq = {}
        self._task_proto_counts = {}

    @torch.no_grad()
    def _compute_universal_class_statistics(self, task_id):
        for c in self._uni_proto_sums:
            n = self._uni_proto_counts[c]
            mean = self._uni_proto_sums[c] / n
            var = (self._uni_proto_sum_sq[c] / n) - (mean * mean)
            self.uni_cls_mean[c] = mean
            self.uni_cls_cov[c] = self._shrink_var(var, n)
        if len(self._uni_proto_sums) > 0:
            print(f"[Task {task_id}] Computed universal statistics for {len(self._uni_proto_sums)} classes (total: {len(self.uni_cls_mean)})")
        self._uni_proto_sums = {}
        self._uni_proto_sum_sq = {}
        self._uni_proto_counts = {}

    def _classifier_align(self):
        K = len(self.cls_mean)
        if K == 0:
            return
        print(f"[CA] Retraining head with {K} classes for {self.ca_epochs} epochs (balanced, {self.ca_samples} per class)")

        optimizer = torch.optim.SGD(
            self.model.head.parameters(),
            lr=self.ca_lr, momentum=0.9, weight_decay=0,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.ca_epochs)

        all_classes = sorted(self.cls_mean.keys())
        D = next(iter(self.cls_mean.values())).shape[-1]

        means = torch.empty(K, D, device=self.device)
        stds = torch.empty(K, D, device=self.device)
        for i, c in enumerate(all_classes):
            if c in self.uni_cls_mean:
                means[i] = self.uni_cls_mean[c]
                stds[i] = self.uni_cls_cov[c].sqrt()
            else:
                means[i] = self.cls_mean[c]
                stds[i] = self.cls_cov[c].sqrt()
        class_ids = torch.tensor(all_classes, device=self.device)

        bs = self.batchsize * 5

        for epoch in range(self.ca_epochs):
            order = torch.arange(K, device=self.device).repeat(self.ca_samples)
            order = order[torch.randperm(order.shape[0], device=self.device)]

            total_loss = 0.0
            for start in range(0, order.shape[0], bs):
                cls_idx = order[start:start + bs]
                eps = torch.randn(cls_idx.shape[0], D, device=self.device)
                feats = means[cls_idx] + eps * stds[cls_idx]
                labels = class_ids[cls_idx]

                logits = self.model.head(feats)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            print(f"  CA Epoch {epoch+1}/{self.ca_epochs} Loss: {total_loss:.4f}")

    def train_learner(self):
        eval_dict = dict()
        self.model.eval()
        for images, labels, idx in self.train_dataloader:
            self.samples_cnt += images.size(0)
            result = self.online_step(images, labels, idx)
            self.report_training(self.samples_cnt, result["acc"])
            self._wandb.log({
                "train/loss_cls": result["loss_cls"],
                "train/acc": result["acc"],
                "train/lr": self.optimizer.param_groups[0]["lr"],
            }, step=self.samples_cnt)

            if self.samples_cnt > self.num_eval:
                with torch.no_grad():
                    test_sampler = OnlineTestSampler(self.test_dataset, self.exposed_classes)
                    test_dataloader = DataLoader(
                        self.test_dataset, batch_size=self.batchsize * 8, shuffle=False,
                        sampler=test_sampler, num_workers=self.n_worker,
                    )
                    eval_dict = self.online_evaluate(test_dataloader)
                    self.eval_results["test_acc"].append(eval_dict["avg_acc"])
                    self.eval_results["avg_acc"].append(eval_dict["cls_acc"])
                    self.eval_results["data_cnt"].append(self.samples_cnt)
                    self.report_test(self.samples_cnt, eval_dict["avg_acc"])
                    n_eval_pts = len(self.eval_results["test_acc"])
                    a_avg_running = float(sum(self.eval_results["test_acc"]) / max(1, n_eval_pts))
                    self._wandb.log({
                        "eval/test_acc": eval_dict["avg_acc"],
                        "eval/A_last": eval_dict["avg_acc"],
                        "eval/A_avg_running": a_avg_running,
                        "eval/task_id": self._cur_task_id,
                        "eval/num_classes": len(self.exposed_classes),
                        "eval/samples": self.samples_cnt,
                    }, step=self.samples_cnt)
                    self.num_eval += self.eval_period
            sys.stdout.flush()
        if len(eval_dict) != 0:
            self.report_test(self.samples_cnt, eval_dict["avg_acc"])

    def online_step(self, X, y, idx):
        self.add_new_class(y)
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        X = X.to(self.device)
        y = y.to(self.device)

        _acc, _iter = 0.0, 0
        _loss_cls = 0.0
        for _ in range(int(self.online_iter)):
            self.model.train()
            feat = self.model.features(X)
            logits = self.model.head(feat)
            loss_cls = F.cross_entropy(logits, y)

            self.optimizer.zero_grad()
            loss_cls.backward()
            self.optimizer.step()

            self._scheduler_step_count += 1
            if "cos" in self.sched_name:
                self.scheduler.step()

            with torch.no_grad():
                self.model.eval()
                feat = self.model.features(X)
                self._accumulate_features(feat, y)
                if self.model.universal_adapter is not None:
                    feat_uni = self.model.features_universal(X)
                    self._accumulate_universal_features(feat_uni, y)
                logits = self.model.head(feat)
                _, pred_label = torch.max(logits, 1)
                acc = (pred_label == y).sum().item() / y.size(0)
            _acc += acc
            _iter += 1
            _loss_cls += loss_cls.item()

        return {
            "acc": _acc / _iter,
            "loss_cls": _loss_cls / _iter,
        }

    @torch.no_grad()
    def _route_logits(self, x):
        n_adapters = len(self.model.task_adapters)
        if n_adapters == 0:
            return self.model(x)

        task_logits = []
        for idx in range(n_adapters):
            feat = self.model.features_with_task(x, idx)
            task_logits.append(self.model.head(feat))
        task_stack = torch.stack(task_logits, dim=0)

        probs = F.softmax(task_stack, dim=-1)
        log_probs = F.log_softmax(task_stack, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        best_t = entropy.argmin(dim=0)
        bs = x.size(0)
        selected = task_stack[best_t, torch.arange(bs, device=x.device)]

        if self.model.universal_adapter is not None:
            feat_uni = self.model.features_universal(x)
            uni_logits = self.model.head(feat_uni)
            return F.softmax(selected, dim=-1) + F.softmax(uni_logits, dim=-1)
        return selected

    def online_evaluate(self, test_loader):
        total_correct = 0.0
        total_num_data = 0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        self.model.eval()

        with torch.no_grad():
            for data in test_loader:
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                logit = self._route_logits(x)
                _, pred_label = logit.topk(self.topk, 1, True, True)

                total_correct += torch.sum(pred_label == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred_label[:, 0])
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

        avg_acc = total_correct / total_num_data
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        return {"avg_acc": avg_acc, "cls_acc": cls_acc}

    def report_training(self, sample_num, train_acc):
        print(
            f"Train | Sample # {sample_num} | train_acc {train_acc:.4f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples - sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, avg_acc):
        print(f"Test | Sample # {sample_num} | test_acc {avg_acc:.4f} | ")
