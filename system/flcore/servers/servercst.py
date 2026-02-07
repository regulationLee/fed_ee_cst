import os
import re
import time
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientcst import clientGH
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from torch.utils.data import DataLoader


class FedGH(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientGH)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.CEloss = nn.CrossEntropyLoss()
        self.server_learning_rate = args.server_learning_rate
        self.server_epochs = args.server_epochs
        self.stage1_mode, self.stage4_mode = self._resolve_cst_modes(args)
        self.n_neg = self._get_env_int("CST_N_NEG", int(getattr(args, "cst_n_neg", 1)))
        self.n_neg = max(0, self.n_neg)
        self.class_proto_lists = {}

        head = load_item(self.clients[0].role, 'model', self.clients[0].save_folder_name).head
        save_item(head, 'Server', 'head', self.save_folder_name)


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_parameters()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()
                client.collect_protos()

            self.receive_protos()
            self._log_round(i)
            self.train_head()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()


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
                if token in {"4_0", "4_1", "4_2"}:
                    stage4 = token
        return stage1, stage4

    def _get_env_int(self, name, default):
        val = os.getenv(name, "").strip()
        if val == "":
            return default
        try:
            return int(val)
        except ValueError:
            return default

    def _to_tensor(self, value):
        if torch.is_tensor(value):
            return value.to(self.device)
        return torch.tensor(value, device=self.device)

    def _log_round(self, round_id):
        print(
            f"[CST] round={round_id} stage1={self.stage1_mode} stage4={self.stage4_mode} "
            f"proto_classes={len(self.class_proto_lists)} n_neg={self.n_neg}"
        )

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        uploaded_protos = []
        class_proto_lists = {}
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            protos = load_item(client.role, 'protos', client.save_folder_name)
            for cc, proto in protos.items():
                cc_int = int(cc)
                y = torch.tensor(cc_int, dtype=torch.int64, device=self.device)
                uploaded_protos.append((proto, y))
                proto_t = self._to_tensor(proto).detach()
                class_proto_lists.setdefault(cc_int, []).append(proto_t)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        save_item(uploaded_protos, self.role, 'uploaded_protos', self.save_folder_name)
        self.class_proto_lists = class_proto_lists
    
    def train_head(self):
        head = load_item('Server', 'head', self.save_folder_name).to(self.device)
        head.train()

        if self.stage1_mode == "1_0":
            uploaded_protos = load_item(self.role, 'uploaded_protos', self.save_folder_name)
            if not uploaded_protos:
                return
            proto_loader = DataLoader(uploaded_protos, self.batch_size, drop_last=False, shuffle=True)
            opt_h = torch.optim.SGD(head.parameters(), lr=self.server_learning_rate)
            for _ in range(self.server_epochs):
                for p, y in proto_loader:
                    p = p.to(self.device)
                    y = y.to(self.device)
                    out = head(p)
                    loss = self.CEloss(out, y)
                    opt_h.zero_grad()
                    loss.backward()
                    opt_h.step()
            save_item(head, 'Server', 'head', self.save_folder_name)
            return

        if len(self.class_proto_lists) == 0:
            return
        train_proto_lists = self.class_proto_lists
        available_classes = sorted(train_proto_lists.keys())
        if len(available_classes) == 0:
            return
        self._log_proto_usage(train_proto_lists)

        if self.stage1_mode == "1_2":
            print(f"[CST] stage1_2 training classes={len(available_classes)}")
            base_head = copy.deepcopy(head)
            new_weight = base_head.weight.detach().clone()
            for idx, k in enumerate(available_classes):
                if idx % 10 == 0 or idx + 1 == len(available_classes):
                    print(f"[CST] stage1_2 progress {idx+1}/{len(available_classes)}")
                head_k = copy.deepcopy(base_head)
                head_k.train()
                opt_k = torch.optim.SGD(head_k.parameters(), lr=self.server_learning_rate)
                for _ in range(self.server_epochs):
                    for proto in train_proto_lists.get(k, []):
                        loss = self._positive_only_loss(head_k, k, proto)
                        opt_k.zero_grad()
                        loss.backward()
                        opt_k.step()
                new_weight[k] = head_k.weight[k].detach()
            head.weight.data.copy_(new_weight)
            save_item(head, 'Server', 'head', self.save_folder_name)
            return

        if self.stage1_mode == "1_1":
            print(f"[CST] stage1_1 training classes={len(available_classes)} n_neg={self.n_neg}")
            new_weight, new_bias = self._train_stage1_1_binary_heads(train_proto_lists, head)
            head.weight.data.copy_(new_weight)
            if head.bias is not None and new_bias is not None:
                head.bias.data.copy_(new_bias)
            save_item(head, 'Server', 'head', self.save_folder_name)
            return

    def _positive_only_loss(self, head, class_id, pos_proto):
        if not torch.is_tensor(pos_proto):
            pos_proto = self._to_tensor(pos_proto)
        pos_proto = pos_proto.to(self.device)
        weight = head.weight
        pos_logit = torch.dot(weight[class_id], pos_proto)
        if head.bias is not None:
            pos_logit = pos_logit + head.bias[class_id]
        return F.softplus(-pos_logit)

    def _train_stage1_1_binary_heads(self, train_proto_lists, base_head):
        available_classes = sorted(train_proto_lists.keys())
        in_dim = int(base_head.weight.shape[1])
        new_weight = base_head.weight.detach().clone()
        new_bias = base_head.bias.detach().clone() if base_head.bias is not None else None
        for idx, k in enumerate(available_classes):
            if idx % 10 == 0 or idx + 1 == len(available_classes):
                print(f"[CST] stage1_1 progress {idx+1}/{len(available_classes)}")
            head_bin = nn.Linear(in_dim, 2, bias=base_head.bias is not None).to(self.device)
            with torch.no_grad():
                head_bin.weight.zero_()
                head_bin.weight[1].copy_(base_head.weight[k].detach())
                if head_bin.bias is not None:
                    head_bin.bias.zero_()
                    if base_head.bias is not None:
                        head_bin.bias[1].copy_(base_head.bias[k].detach())
            opt_k = torch.optim.SGD(head_bin.parameters(), lr=self.server_learning_rate)
            for _ in range(self.server_epochs):
                for proto in train_proto_lists.get(k, []):
                    neg_protos = self._sample_negative_protos(k, available_classes, train_proto_lists)
                    loss = self._binary_head_loss(head_bin, proto, neg_protos)
                    opt_k.zero_grad()
                    loss.backward()
                    opt_k.step()
            new_weight[k] = head_bin.weight[1].detach()
            if new_bias is not None and head_bin.bias is not None:
                new_bias[k] = head_bin.bias[1].detach()
        return new_weight, new_bias

    def _binary_head_loss(self, head_bin, pos_proto, neg_protos):
        pos_proto = self._to_tensor(pos_proto)
        proto_list = [pos_proto] + [self._to_tensor(p) for p in neg_protos]
        labels = torch.tensor([1] + [0] * len(neg_protos), device=self.device, dtype=torch.long)
        protos = torch.stack(proto_list, dim=0).to(self.device)
        logits = head_bin(protos)
        return F.cross_entropy(logits, labels)

    def _sample_negative_protos(self, class_id, available_classes, train_proto_lists):
        neg_classes = [
            c for c in available_classes
            if c != class_id and len(train_proto_lists.get(c, [])) > 0
        ]
        if len(neg_classes) == 0:
            return []
        neg_protos = []
        for _ in range(self.n_neg):
            j = random.choice(neg_classes)
            plist = train_proto_lists[j]
            neg_protos.append(plist[random.randrange(len(plist))])
        return neg_protos

    def _log_proto_usage(self, train_proto_lists):
        counts = [len(v) for v in train_proto_lists.values()]
        if len(counts) == 0:
            return
        total = int(sum(counts))
        avg = float(np.mean(counts)) if counts else 0.0
        min_cnt = int(min(counts))
        max_cnt = int(max(counts))
        print(
            f"[CST] proto_usage total={total} "
            f"per_class_avg={avg:.2f} min={min_cnt} max={max_cnt}"
        )
