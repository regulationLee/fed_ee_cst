import time
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flcore.clients.clientghee import clientGH
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item


class FedGH(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientGH)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.CEloss = nn.CrossEntropyLoss()
        self.server_learning_rate = args.server_learning_rate
        self.server_epochs = args.server_epochs
        self.reinit_head = bool(getattr(args, "ghee_reinit_head", False))

        model = load_item(self.clients[0].role, 'model', self.clients[0].save_folder_name)
        head = getattr(model, "head", None)
        if head is not None:
            save_item(head, 'Server', 'head', self.save_folder_name)

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_parameters()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()
                client.collect_protos()

            self.receive_protos()
            if getattr(self, "last_proto_counts", None) is not None:
                parts = []
                for eid in sorted(self.last_proto_counts.keys()):
                    parts.append(f"Exit{eid}: {self.last_proto_counts[eid]}")
                counts_str = ", ".join(parts) if parts else "none"
                print(f"Proto usage this round - total: {self.last_proto_total} ({counts_str})")
            self.train_head()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

    def _to_tensor(self, value):
        if torch.is_tensor(value):
            return value
        return torch.tensor(value)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        exit_proto_lists = {}
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            protos_by_exit = load_item(client.role, 'protos', client.save_folder_name)
            if not protos_by_exit:
                continue
            for exit_id, class_dict in protos_by_exit.items():
                for cc, proto in class_dict.items():
                    cc_int = int(cc)
                    proto_t = self._to_tensor(proto).detach().cpu()
                    exit_proto_lists.setdefault(exit_id, {}).setdefault(cc_int, []).append(proto_t)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples if tot_samples > 0 else 0.0

        uploaded_protos = []
        for class_map in exit_proto_lists.values():
            for cc, proto_list in class_map.items():
                y = torch.tensor(cc, dtype=torch.int64)
                for proto in proto_list:
                    uploaded_protos.append((proto, y))
        self.uploaded_protos = uploaded_protos
        self.last_proto_counts = {}
        for exit_id, class_map in exit_proto_lists.items():
            self.last_proto_counts[exit_id] = sum(len(v) for v in class_map.values())
        self.last_proto_total = sum(self.last_proto_counts.values()) if self.last_proto_counts else 0

    def train_head(self):
        if not getattr(self, "uploaded_protos", None):
            return

        first_proto = self.uploaded_protos[0][0]
        in_dim = int(first_proto.numel())
        head = None
        if not self.reinit_head:
            head = load_item('Server', 'head', self.save_folder_name)
        if head is None or not isinstance(head, nn.Linear) or head.in_features != in_dim:
            head = nn.Linear(in_dim, self.num_classes)
        head = head.to(self.device)

        proto_loader = DataLoader(self.uploaded_protos, self.batch_size, drop_last=False, shuffle=True)
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

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        per_exit_correct_list = []
        for c in self.clients:
            result = c.test_metrics()
            per_exit_correct = None
            if isinstance(result, tuple) and len(result) == 5:
                ct, ns, auc, _, per_exit_correct = result
            elif isinstance(result, tuple) and len(result) == 4:
                ct, ns, auc, _ = result
            else:
                ct, ns, auc = result
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
            if per_exit_correct:
                exit_parts = []
                for eid in sorted(per_exit_correct.keys()):
                    exit_parts.append(f"Exit{eid}: {per_exit_correct[eid] * 1.0 / ns}")
                exit_str = ", ".join(exit_parts)
                print(f'Client {c.id}: Acc Final: {ct*1.0/ns}, {exit_str}, AUC: {auc}')
            else:
                print(f'Client {c.id}: Acc: {ct*1.0/ns}, AUC: {auc}')
            per_exit_correct_list.append(per_exit_correct)

        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct, tot_auc, per_exit_correct_list

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc is None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        self.rs_test_auc.append(test_auc)

        per_exit_correct_list = stats[4] if len(stats) > 4 else []
        if per_exit_correct_list:
            per_exit_totals = {}
            per_exit_samples = {}
            for ns, per_exit in zip(stats[1], per_exit_correct_list):
                if not per_exit:
                    continue
                for eid, correct in per_exit.items():
                    per_exit_totals[eid] = per_exit_totals.get(eid, 0.0) + correct
                    per_exit_samples[eid] = per_exit_samples.get(eid, 0.0) + ns

            print("Final Averaged Test Accuracy: {:.4f}".format(test_acc))
            if per_exit_totals:
                final_exit = max(per_exit_totals.keys())
                for eid in sorted(per_exit_totals.keys()):
                    if eid == final_exit:
                        continue
                    denom = per_exit_samples.get(eid, 0.0)
                    if denom > 0:
                        exit_acc = per_exit_totals[eid] * 1.0 / denom
                        print("Exit{} Averaged Test Accuracy: {:.4f}".format(eid, exit_acc))
        else:
            print("Averaged Test Accuracy: {:.4f}".format(test_acc))

        if loss is not None:
            loss.append(test_acc)
