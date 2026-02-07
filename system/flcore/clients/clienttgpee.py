import copy
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from flcore.clients.clientbase import Client, load_item, save_item
from flcore.models.early_exit_wrappers import wrap_with_early_exit


class clientTGP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.args = args
        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda
        self.num_exits = 2
        self.ee_weights = self._resolve_ee_weights()
        self.ee_adapt = bool(getattr(args, "ee_adapt", False))
        self.ee_adapt_every = int(getattr(args, "ee_adapt_every", 1))
        self.ee_adapt_step = float(getattr(args, "ee_adapt_step", 0.1))
        self.ee_min_w = float(getattr(args, "ee_min_w", 0.2))
        self.ee_max_w = float(getattr(args, "ee_max_w", 0.8))
        self.ee_w_final = self.ee_weights[-1] if len(self.ee_weights) == 2 else 0.5
        self._printed_ee = False

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

    def _get_backbone_params(self, model):
        if hasattr(model, "backbone"):
            params = list(model.backbone.parameters())
        elif hasattr(model, "base"):
            params = list(model.base.parameters())
        else:
            params = list(model.parameters())
        return [p for p in params if p.requires_grad]

    def _update_adaptive_weights(self, loss_early, loss_final, model):
        if not self.ee_adapt:
            return
        params = self._get_backbone_params(model)
        if not params:
            return
        g_early = torch.autograd.grad(loss_early, params, retain_graph=True, allow_unused=True)
        g_final = torch.autograd.grad(loss_final, params, retain_graph=True, allow_unused=True)

        dot = 0.0
        norm_e = 0.0
        norm_f = 0.0
        for ge, gf in zip(g_early, g_final):
            if ge is None or gf is None:
                continue
            dot = dot + torch.sum(ge * gf)
            norm_e = norm_e + torch.sum(ge * ge)
            norm_f = norm_f + torch.sum(gf * gf)
        if norm_e.item() <= 0 or norm_f.item() <= 0:
            return
        cos = (dot / (torch.sqrt(norm_e) * torch.sqrt(norm_f) + 1e-12)).item()

        min_w = min(self.ee_min_w, self.ee_max_w)
        max_w = max(self.ee_min_w, self.ee_max_w)
        w_final = self.ee_w_final + self.ee_adapt_step * (-cos)
        w_final = max(min_w, min(max_w, w_final))
        self.ee_w_final = w_final
        self.ee_weights = [1.0 - w_final, w_final]

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

    def _forward_exits(self, model, x):
        if hasattr(model, "get_exit_features"):
            feats_by_exit = model.get_exit_features(x)
        else:
            feats_by_exit = model.extract_exit_features(x)
        if hasattr(model, "_ensure_heads_initialized"):
            model._ensure_heads_initialized(feats_by_exit)
        logits_by_exit = {}
        for exit_id, feats in feats_by_exit.items():
            head = model.head_by_exit[str(exit_id)]
            logits_by_exit[exit_id] = head(feats)
        return feats_by_exit, logits_by_exit

    def train(self):
        trainloader = self.load_train_data()
        model = self._load_wrapped_model()
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        model.train()

        if not self._printed_ee:
            if self.ee_adapt:
                print(
                    f"[Client {self.id}] ee_weights={self.ee_weights} lamda={self.lamda} "
                    f"adapt_every_epoch={self.ee_adapt_every} adapt_step={self.ee_adapt_step} "
                    f"w_final_range=({self.ee_min_w},{self.ee_max_w})"
                )
            else:
                print(f"[Client {self.id}] ee_weights={self.ee_weights} lamda={self.lamda} (fixed)")
            self._printed_ee = True

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        protos_by_exit = defaultdict(lambda: defaultdict(list))
        num_batches = len(trainloader)
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                feats_by_exit, logits_by_exit = self._forward_exits(model, x)
                exit_ids = sorted(feats_by_exit.keys())
                if not exit_ids:
                    continue

                early_exit = exit_ids[0]
                final_exit = exit_ids[-1]

                loss_early = self.loss(logits_by_exit[early_exit], y)
                loss_final_ce = self.loss(logits_by_exit[final_exit], y)
                loss_final_proto = 0.0
                if global_protos is not None:
                    proto_new = copy.deepcopy(feats_by_exit[final_exit].detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if y_c in global_protos and type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    loss_final_proto = self.loss_mse(proto_new, feats_by_exit[final_exit]) * self.lamda

                loss_final = loss_final_ce + loss_final_proto

                if (
                    self.ee_adapt
                    and self.ee_adapt_every > 0
                    and ((epoch + 1) % self.ee_adapt_every == 0)
                    and (i + 1 == num_batches)
                ):
                    self._update_adaptive_weights(loss_early, loss_final, model)

                weights = self._get_weight_tensor(y.device, len(exit_ids))
                loss = 0.0
                for idx, eid in enumerate(exit_ids):
                    if eid == early_exit:
                        loss = loss + weights[idx] * loss_early
                    elif eid == final_exit:
                        loss = loss + weights[idx] * loss_final

                for exit_id, feats in feats_by_exit.items():
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        protos_by_exit[exit_id][y_c].append(feats[i, :].detach().data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        final_protos_by_exit = {}
        for exit_id, class_map in protos_by_exit.items():
            final_protos_by_exit[exit_id] = agg_func(class_map)

        save_item(final_protos_by_exit, self.role, 'protos', self.save_folder_name)
        save_item(model, self.role, 'model', self.save_folder_name)
        print(f"[Client {self.id}] ee_weights_end={self.ee_weights} adaptive={self.ee_adapt}")

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics(self):
        testloader = self.load_test_data()
        model = self._load_wrapped_model()
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        model.eval()

        test_acc = 0
        test_acc_exit0 = 0
        test_num = 0
        per_exit_correct = defaultdict(int)

        if global_protos is not None:
            with torch.no_grad():
                for x, y in testloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    feats_by_exit, logits_by_exit = self._forward_exits(model, x)
                    exit_ids = sorted(feats_by_exit.keys())
                    if not exit_ids:
                        continue

                    early_exit = exit_ids[0]
                    final_exit = exit_ids[-1]

                    early_logits = logits_by_exit.get(early_exit)
                    if early_logits is not None:
                        early_correct = (torch.sum(torch.argmax(early_logits, dim=1) == y)).item()
                        per_exit_correct[early_exit] += early_correct
                        test_acc_exit0 += early_correct

                    output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
                    final_feats = feats_by_exit[final_exit]
                    for i, r in enumerate(final_feats):
                        for j, pro in global_protos.items():
                            if type(pro) != type([]):
                                output[i, j] = self.loss_mse(r, pro)

                    final_correct = (torch.sum(torch.argmin(output, dim=1) == y)).item()
                    per_exit_correct[final_exit] += final_correct
                    test_acc += final_correct
                    test_num += y.shape[0]

            return test_acc, test_num, 0, test_acc_exit0, dict(per_exit_correct)
        return 0, 1e-5, 0, 0, {}

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = self._load_wrapped_model()
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
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

                feats_by_exit, logits_by_exit = self._forward_exits(model, x)
                exit_ids = sorted(feats_by_exit.keys())
                if not exit_ids:
                    continue

                early_exit = exit_ids[0]
                final_exit = exit_ids[-1]

                loss_early = self.loss(logits_by_exit[early_exit], y)
                loss_final_ce = self.loss(logits_by_exit[final_exit], y)
                loss_final_proto = 0.0
                if global_protos is not None:
                    proto_new = copy.deepcopy(feats_by_exit[final_exit].detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if y_c in global_protos and type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    loss_final_proto = self.loss_mse(proto_new, feats_by_exit[final_exit]) * self.lamda

                loss_final = loss_final_ce + loss_final_proto
                weights = self._get_weight_tensor(y.device, len(exit_ids))
                loss = 0.0
                for idx, eid in enumerate(exit_ids):
                    if eid == early_exit:
                        loss = loss + weights[idx] * loss_early
                    elif eid == final_exit:
                        loss = loss + weights[idx] * loss_final

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