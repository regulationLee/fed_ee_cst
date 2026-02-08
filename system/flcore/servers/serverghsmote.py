import os
import re
import time
import random
import torch
import torch.nn as nn
from collections import defaultdict
from flcore.clients.clientghsmote import clientGH
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from torch.utils.data import DataLoader


class FedGH(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientGH)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.CEloss = nn.CrossEntropyLoss()
        self.server_learning_rate = args.server_learning_rate
        self.server_epochs = args.server_epochs
        self.stage1_mode = self._resolve_ghsmote_stage(args)

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

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_protos()
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


    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        # Collect one prototype per class per client.
        # We later apply a SMOTE-style augmentation (or balanced sampling) on the server.
        class_proto_lists = defaultdict(list)
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            protos = load_item(client.role, 'protos', client.save_folder_name)
            for cc in protos.keys():
                class_proto_lists[int(cc)].append(protos[cc])
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        uploaded_protos = self._build_smote_protos(class_proto_lists)
        save_item(uploaded_protos, self.role, 'uploaded_protos', self.save_folder_name)

    def _resolve_ghsmote_stage(self, args):
        # Same stage naming as FedGHEE/FedTGPEE: 1_0 (default), 1_1 (min-cap), 1_2 (max-cap)
        stage1 = "1_0"
        raw_vals = [
            os.getenv("GHSMOTE_STAGE", "").strip().lower(),
            str(getattr(args, "ghsmote_stage", "")).strip().lower(),
        ]
        for raw in raw_vals:
            if not raw:
                continue
            tokens = [t for t in re.split(r"[+,|\\s]+", raw) if t]
            for token in tokens:
                token = token.replace("stage", "").replace("ghsmote", "").strip()
                if token in {"1_0", "1_1", "1_2"}:
                    stage1 = token
        return stage1

    def _build_smote_protos(self, class_proto_lists):
        """Create same number of synthetic prototypes per class (doubling total).
        If stage1_mode is 1_1 or 1_2, apply min/max cap sampling (same as FedGHEE/FedTGPEE).
        """
        if self.stage1_mode == "1_0":
            return self._double_with_smote(class_proto_lists)
        return self._build_balanced_smote_protos(class_proto_lists)

    def _double_with_smote(self, class_proto_lists):
        uploaded_protos = []
        for cc, proto_list in class_proto_lists.items():
            if not proto_list:
                continue
            y = torch.tensor(cc, dtype=torch.int64, device=self.device)
            # Add real prototypes.
            for proto in proto_list:
                uploaded_protos.append((proto, y))
            # Add SMOTE-style synthetic prototypes (same count as real).
            synth_list = self._smote_augment(proto_list, len(proto_list))
            for proto in synth_list:
                uploaded_protos.append((proto, y))
        return uploaded_protos

    def _build_balanced_smote_protos(self, class_proto_lists):
        # Compute caps based on real (base) prototypes only.
        base_counts = {cc: len(plist) for cc, plist in class_proto_lists.items() if len(plist) > 0}
        if not base_counts:
            return []
        min_base = min(base_counts.values())
        max_base = max(base_counts.values())
        if self.stage1_mode == "1_1":
            cap = 2 * min_base
            print(f"[GHSMOTE] stage1_1 min_base={min_base} cap={cap}")
        else:
            cap = max_base
            print(f"[GHSMOTE] stage1_2 max_base={max_base} cap={cap}")

        uploaded_protos = []
        for cc, base_list in class_proto_lists.items():
            if not base_list:
                continue
            # Synthetic list is same size as base list (SMOTE doubling).
            smote_list = self._smote_augment(base_list, len(base_list))
            picked = self._pick_with_cap(base_list, smote_list, cap)
            y = torch.tensor(cc, dtype=torch.int64, device=self.device)
            for proto in picked:
                uploaded_protos.append((proto, y))
        return uploaded_protos

    def _pick_with_cap(self, base_list, smote_list, cap):
        # Same rule as FedGHEE/FedTGPEE: use base first, then SMOTE to fill; downsample base if too many.
        if cap <= 0:
            return []
        base_list = list(base_list)
        smote_list = list(smote_list)
        base_count = len(base_list)
        if base_count >= cap:
            if base_count == cap:
                return base_list
            return random.sample(base_list, cap)
        remaining = cap - base_count
        if remaining <= 0:
            return base_list
        if len(smote_list) <= remaining:
            return base_list + smote_list
        return base_list + random.sample(smote_list, remaining)

    def _smote_augment(self, proto_list, num_new):
        """Simple SMOTE in feature space: interpolate between two random prototypes."""
        if num_new <= 0:
            return []
        if len(proto_list) == 1:
            # Only one prototype available: duplicate it to keep counts balanced.
            return [proto_list[0].detach().clone() for _ in range(num_new)]
        synth = []
        for _ in range(num_new):
            a, b = random.sample(proto_list, 2)
            lam = torch.rand(1, device=a.device)
            proto = a + lam * (b - a)
            synth.append(proto.detach())
        return synth
    
    def train_head(self):
        uploaded_protos = load_item(self.role, 'uploaded_protos', self.save_folder_name)
        proto_loader = DataLoader(uploaded_protos, self.batch_size, drop_last=False, shuffle=True)
        head = load_item('Server', 'head', self.save_folder_name)
        
        opt_h = torch.optim.SGD(head.parameters(), lr=self.server_learning_rate)

        for _ in range(self.server_epochs):
            for p, y in proto_loader:
                out = head(p)
                loss = self.CEloss(out, y)
                opt_h.zero_grad()
                loss.backward()
                opt_h.step()

        save_item(head, 'Server', 'head', self.save_folder_name)
