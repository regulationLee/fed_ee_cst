import os
import time
from collections import defaultdict

import numpy as np
import torch
from sklearn.preprocessing import label_binarize
from sklearn import metrics

from flcore.clients.clientbase import Client, load_item, save_item
from flcore.models.early_exit_wrappers import wrap_with_early_exit


class clientGH(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        self.args = args
        self.num_exits = 2
        self.ee_weights = self._resolve_ee_weights()
        self.proto_norm = getattr(args, "proto_norm", "none")

    def _resolve_ee_weights(self):
        env = os.getenv("EE_WEIGHTS", "").strip()
        if not env:
            env = os.getenv("CWGH_EE_WEIGHTS", "").strip()
        weights = self._parse_weights(env)
        if not weights:
            arg_val = getattr(self.args, "ee_weights", None)
            weights = self._parse_weights(arg_val)
        if not weights:
            weights = [0.5, 0.5]
        if len(weights) != self.num_exits:
            weights = [1.0 / self.num_exits] * self.num_exits
        return weights

    def _parse_weights(self, value):
        if not value:
            return None
        parts = []
        for token in str(value).replace(";", ",").replace(" ", ",").split(","):
            token = token.strip()
            if token == "":
                continue
            try:
                parts.append(float(token))
            except ValueError:
                continue
        return parts or None

    def _get_weight_tensor(self, device, num_exits):
        weights = self.ee_weights
        if len(weights) != num_exits:
            weights = [1.0 / num_exits] * num_exits
        w = torch.tensor(weights, device=device, dtype=torch.float32)
        s = float(w.sum().item())
        if s <= 0:
            w = torch.ones(num_exits, device=device) / float(num_exits)
        else:
            w = w / s
        return w

    def _load_wrapped_model(self):
        model = load_item(self.role, 'model', self.save_folder_name)
        if model is None:
            raise RuntimeError(f"Model not found for {self.role} at {self.save_folder_name}")
        model = wrap_with_early_exit(
            model,
            None,
            self.num_classes,
            feature_dim=getattr(self.args, "feature_dim", None),
            num_exits=self.num_exits,
        )
        return model.to(self.device)

    def _apply_shared_head(self, model, shared_head):
        if shared_head is None:
            return
        payload = {
            "in_dim": shared_head.in_features,
            "state_dict": shared_head.state_dict(),
        }
        final_exit = self.num_exits - 1
        model.load_exit_heads({final_exit: payload})

    def _weighted_loss(self, logits_by_exit, y):
        exit_ids = sorted(logits_by_exit.keys())
        weights = self._get_weight_tensor(y.device, len(exit_ids))
        loss = 0.0
        for i, eid in enumerate(exit_ids):
            loss = loss + weights[i] * self.loss(logits_by_exit[eid], y)
        return loss

    def train(self):
        trainloader = self.load_train_data()
        model = self._load_wrapped_model()
        shared_head = load_item('Server', 'head', self.save_folder_name)
        self._apply_shared_head(model, shared_head)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for _ in range(max_local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                logits_by_exit = None
                if hasattr(model, "forward_all_exits"):
                    logits_by_exit = model.forward_all_exits(x)
                if logits_by_exit:
                    loss = self._weighted_loss(logits_by_exit, y)
                else:
                    output = model(x)
                    loss = self.loss(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        save_item(model, self.role, 'model', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self):
        model = self._load_wrapped_model()
        shared_head = load_item('Server', 'head', self.save_folder_name)
        self._apply_shared_head(model, shared_head)
        save_item(model, self.role, 'model', self.save_folder_name)

    def collect_protos(self):
        trainloader = self.load_train_data()
        model = self._load_wrapped_model()
        model.eval()
        protos_by_exit = self._compute_protos_by_exit(model, trainloader)
        save_item(protos_by_exit, self.role, 'protos', self.save_folder_name)

    def _compute_protos_by_exit(self, model, loader):
        eps = 1e-12
        sum_feats = defaultdict(dict)
        counts = defaultdict(lambda: defaultdict(int))
        with torch.no_grad():
            for x, y in loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                if hasattr(model, "get_exit_features"):
                    feats_by_exit = model.get_exit_features(x)
                else:
                    feats_by_exit = model.extract_exit_features(x)
                for exit_id, feats in feats_by_exit.items():
                    if self.proto_norm == "l2":
                        feats = feats / (feats.norm(dim=1, keepdim=True) + eps)
                    feats_cpu = feats.detach().cpu()
                    for j, yy in enumerate(y):
                        lbl = int(yy.item())
                        if lbl not in sum_feats[exit_id]:
                            sum_feats[exit_id][lbl] = torch.zeros_like(feats_cpu[j])
                        sum_feats[exit_id][lbl] += feats_cpu[j]
                        counts[exit_id][lbl] += 1
        protos_by_exit = {}
        for exit_id, class_sums in sum_feats.items():
            protos_by_exit[exit_id] = {}
            for lbl, s in class_sums.items():
                proto = s / counts[exit_id][lbl]
                if self.proto_norm == "l2":
                    proto = proto / (torch.norm(proto, p=2) + eps)
                protos_by_exit[exit_id][lbl] = proto
        return protos_by_exit

    def test_metrics(self):
        testloader = self.load_test_data()
        model = self._load_wrapped_model()
        shared_head = load_item('Server', 'head', self.save_folder_name)
        self._apply_shared_head(model, shared_head)
        model.eval()

        test_acc = 0
        test_num = 0
        per_exit_correct = defaultdict(int)
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                logits_by_exit = None
                if hasattr(model, "forward_all_exits"):
                    logits_by_exit = model.forward_all_exits(x)
                if logits_by_exit:
                    exit_ids = sorted(logits_by_exit.keys())
                    final_exit = exit_ids[-1]
                    for eid, logits in logits_by_exit.items():
                        per_exit_correct[eid] += (torch.sum(torch.argmax(logits, dim=1) == y)).item()
                    final_logits = logits_by_exit[final_exit]
                    test_acc += (torch.sum(torch.argmax(final_logits, dim=1) == y)).item()
                    y_prob.append(final_logits.detach().cpu().numpy())
                else:
                    output = model(x)
                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    y_prob.append(output.detach().cpu().numpy())

                test_num += y.shape[0]
                nc = self.num_classes + 1 if self.num_classes == 2 else self.num_classes
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        auc = 0.0
        if y_prob and y_true:
            try:
                y_prob = np.concatenate(y_prob, axis=0)
                y_true = np.concatenate(y_true, axis=0)
                auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
            except ValueError:
                auc = 0.0

        test_acc_exit0 = 0
        if per_exit_correct:
            early_exit = min(per_exit_correct.keys())
            test_acc_exit0 = per_exit_correct[early_exit]

        return test_acc, test_num, auc, test_acc_exit0, dict(per_exit_correct)

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = self._load_wrapped_model()
        shared_head = load_item('Server', 'head', self.save_folder_name)
        self._apply_shared_head(model, shared_head)
        model.eval()

        train_num = 0
        losses = 0.0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                logits_by_exit = None
                if hasattr(model, "forward_all_exits"):
                    logits_by_exit = model.forward_all_exits(x)
                if logits_by_exit:
                    exit_ids = sorted(logits_by_exit.keys())
                    final_exit = exit_ids[-1]
                    output = logits_by_exit[final_exit]
                else:
                    output = model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return losses, train_num