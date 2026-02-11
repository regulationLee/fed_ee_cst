import os
import re
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from utils.data_utils import read_client_data
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from collections import defaultdict


class clientGH(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # torch.manual_seed(0)
        self.stage1_mode, self.stage4_mode = self._resolve_cst_modes(args)
        self.lambda_gate = self._get_env_float("CST_LAMBDA_GATE", float(getattr(args, "cst_lambda_gate", 10.0)))
        self.eps = 1e-7
        self.class_counts = None
        self.gates = None
        self._freeze_hooks = []
        if self.stage4_mode in {"4_1_1", "4_1_2", "4_1_3"}:
            self.class_counts = self._build_class_counts()
            if self.stage4_mode == "4_1_1":
                self.gates = self._build_gates_hybrid(self.class_counts)
            else:
                self.gates = None

    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        # model.to(self.device)
        model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)
                logits = self._linear_logits(rep, model)
                loss = self.loss(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        save_item(model, self.role, 'model', self.save_folder_name)
        
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        
    def set_parameters(self):
        model = load_item(self.role, 'model', self.save_folder_name)
        if self.stage4_mode in {"4_1_1", "4_1_2", "4_1_3"}:
            global_head = load_item('Server', 'head', self.save_folder_name)
            global_head = global_head.to(self.device)
            global_weight, global_bias = self._get_head_params(global_head)
            local_weight, local_bias = self._get_head_params(model.head)
            if self.class_counts is None:
                self.class_counts = self._build_class_counts()
            if self.stage4_mode == "4_1_1":
                if self.gates is None:
                    self.gates = self._build_gates_hybrid(self.class_counts)
                g = self.gates.to(local_weight.device).unsqueeze(1)
                blended_weight = (1.0 - g) * local_weight + g * global_weight.to(local_weight.device)
                local_weight.data.copy_(blended_weight.data)
                if local_bias is not None and global_bias is not None:
                    gb = global_bias.to(local_bias.device)
                    lb = local_bias
                    blended_bias = (1.0 - self.gates.to(local_bias.device)) * lb + self.gates.to(local_bias.device) * gb
                    local_bias.data.copy_(blended_bias.data)
                if self.gates is not None:
                    g_stats = self.gates
                    print(
                        f"[CST] client={self.id} gate_stats "
                        f"mean={float(g_stats.mean()):.4f} min={float(g_stats.min()):.4f} max={float(g_stats.max()):.4f} "
                        f"stage4={self.stage4_mode}"
                    )
            elif self.stage4_mode == "4_1_2":
                alpha = self._compute_stage4_1_2_alpha(self.class_counts).to(local_weight.device)
                blended_weight = alpha.unsqueeze(1) * local_weight + (1.0 - alpha).unsqueeze(1) * global_weight.to(local_weight.device)
                local_weight.data.copy_(blended_weight.data)
                if local_bias is not None and global_bias is not None:
                    gb = global_bias.to(local_bias.device)
                    lb = local_bias
                    blended_bias = alpha.to(local_bias.device) * lb + (1.0 - alpha.to(local_bias.device)) * gb
                    local_bias.data.copy_(blended_bias.data)
            else:
                alpha = self._compute_stage4_1_2_alpha(self.class_counts).to(local_weight.device)
                local_weight_norm = local_weight / (local_weight.norm(dim=1, keepdim=True) + self.eps)
                global_weight_norm = global_weight.to(local_weight.device)
                global_weight_norm = global_weight_norm / (global_weight_norm.norm(dim=1, keepdim=True) + self.eps)
                blended_weight = alpha.unsqueeze(1) * local_weight_norm + (1.0 - alpha).unsqueeze(1) * global_weight_norm
                local_weight.data.copy_(blended_weight.data)
                if local_bias is not None and global_bias is not None:
                    gb = global_bias.to(local_bias.device)
                    lb = local_bias
                    lb_norm = lb / (lb.abs() + self.eps)
                    gb_norm = gb / (gb.abs() + self.eps)
                    blended_bias = alpha.to(local_bias.device) * lb_norm + (1.0 - alpha.to(local_bias.device)) * gb_norm
                    local_bias.data.copy_(blended_bias.data)
            save_item(model, self.role, 'model', self.save_folder_name)
            return
        head = load_item('Server', 'head', self.save_folder_name)
        for new_param, old_param in zip(head.parameters(), model.head.parameters()):
            old_param.data = new_param.data.clone()
        save_item(model, self.role, 'model', self.save_folder_name)

    def _clear_freeze_hooks(self):
        for hook in getattr(self, "_freeze_hooks", []):
            try:
                hook.remove()
            except Exception:
                pass
        self._freeze_hooks = []

    def _set_frozen_class_mask(self, head, mask_zero):
        self._clear_freeze_hooks()
        if mask_zero is None or not torch.is_tensor(mask_zero) or not mask_zero.any():
            return
        mask = mask_zero.to(head.weight.device)

        def hook_weight(grad):
            if grad is None:
                return grad
            grad = grad.clone()
            grad[mask] = 0
            return grad

        self._freeze_hooks.append(head.weight.register_hook(hook_weight))
        if hasattr(head, "bias") and head.bias is not None:
            def hook_bias(grad):
                if grad is None:
                    return grad
                grad = grad.clone()
                grad[mask] = 0
                return grad
            self._freeze_hooks.append(head.bias.register_hook(hook_bias))

    def collect_protos(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        save_item(agg_func(protos), self.role, 'protos', self.save_folder_name)

    def _resolve_cst_modes(self, args):
        stage1 = "1_0"
        stage4 = "4_0"
        raw_vals = [
            os.getenv("CST_STAGE", "").strip().lower(),
            str(getattr(args, "cst_stage", "")).strip().lower(),
        ]
        for raw in raw_vals:
            if not raw:
                continue
            tokens = [t for t in re.split(r"[+,|\\s]+", raw) if t]
            for token in tokens:
                token = token.replace("stage", "").replace("cst", "").strip()
                if token in {"1_0", "1_1", "1_2"}:
                    stage1 = token
                if token in {"4_0", "4_1_1", "4_1_2", "4_1_3"}:
                    stage4 = token
        return stage1, stage4

    def _get_env_float(self, name, default):
        val = os.getenv(name, "").strip()
        if val == "":
            return default
        try:
            return float(val)
        except ValueError:
            return default

    def _build_class_counts(self):
        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        data_list = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        for _, y in data_list:
            if torch.is_tensor(y):
                label = int(y.item())
            else:
                label = int(y)
            if 0 <= label < self.num_classes:
                counts[label] += 1
        return counts

    def _build_gates_hybrid(self, counts):
        eps = 1e-12
        counts = counts.to(torch.float32)
        total = counts.sum()
        if total <= 0:
            return self.lambda_gate / (counts + self.lambda_gate)
        p = counts / (total + eps)
        u = 1.0 / float(self.num_classes)
        imbalance = torch.clamp((u - p) / (u + eps), min=0.0, max=1.0)
        confidence = self.lambda_gate / (counts + self.lambda_gate)
        return 1.0 - (1.0 - confidence) * (1.0 - imbalance)

    def _compute_stage4_1_2_alpha(self, counts):
        counts = counts.to(torch.float32)
        total = float(counts.sum().item())
        if total <= 0.0:
            return torch.zeros((self.num_classes,), dtype=torch.float32)
        return counts / total

    def _get_head_params(self, head):
        if hasattr(head, "weight"):
            weight = head.weight
            bias = head.bias if hasattr(head, "bias") else None
            return weight, bias
        raise NotImplementedError("CST requires a linear head with a weight matrix.")

    def _linear_logits(self, rep, model):
        weight, bias = self._get_head_params(model.head)
        logits = torch.matmul(rep, weight.t())
        if bias is not None:
            logits = logits + bias
        return logits

    def test_metrics(self):
        testloader = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = model.base(x)
                logits = self._linear_logits(rep, model)
                test_acc += (torch.sum(torch.argmax(logits, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(logits.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = model.base(x)
                logits = self._linear_logits(rep, model)
                loss = self.loss(logits, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos