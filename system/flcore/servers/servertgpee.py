import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flcore.clients.clienttgpee import clientTGP
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item


class FedTGP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientTGP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes

        self.server_learning_rate = args.local_learning_rate
        self.batch_size = args.batch_size
        self.server_epochs = args.server_epochs
        self.margin_threthold = args.margin_threthold

        self.feature_dim = args.feature_dim
        self.server_hidden_dim = self.feature_dim
        
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            TGP = Trainable_Global_Prototypes(
                self.num_classes, 
                self.server_hidden_dim, 
                self.feature_dim, 
                self.device
            ).to(self.device)
            save_item(TGP, self.role, 'TGP', self.save_folder_name)
            print(TGP)
        self.CEloss = nn.CrossEntropyLoss()
        self.MSEloss = nn.MSELoss()

        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        self.min_gap = None
        self.max_gap = None


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_protos()
            self.update_TGP()

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

        self.uploaded_ids = []
        self.uploaded_protos = []
        uploaded_protos_per_client = []
        per_exit_counts = defaultdict(int)
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos_by_exit = load_item(client.role, 'protos', client.save_folder_name)
            if not protos_by_exit:
                continue
            if isinstance(protos_by_exit, dict):
                for exit_id, class_map in protos_by_exit.items():
                    if not class_map:
                        continue
                    for k, proto in class_map.items():
                        self.uploaded_protos.append((proto, k))
                    uploaded_protos_per_client.append(class_map)
                    per_exit_counts[exit_id] += len(class_map)
            else:
                for k in protos_by_exit.keys():
                    self.uploaded_protos.append((protos_by_exit[k], k))
                uploaded_protos_per_client.append(protos_by_exit)

        if per_exit_counts:
            total = sum(per_exit_counts.values())
            exit_str = ", ".join([f"{eid}:{cnt}" for eid, cnt in sorted(per_exit_counts.items())])
            print(f"[FedTGPEE] proto_usage total={total} per_exit=[{exit_str}]")

        # calculate class-wise minimum distance
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        avg_protos = proto_cluster(uploaded_protos_per_client)
        for k1 in avg_protos.keys():
            for k2 in avg_protos.keys():
                if k1 > k2:
                    dis = torch.norm(avg_protos[k1] - avg_protos[k2], p=2)
                    self.gap[k1] = torch.min(self.gap[k1], dis)
                    self.gap[k2] = torch.min(self.gap[k2], dis)
        self.min_gap = torch.min(self.gap)
        for i in range(len(self.gap)):
            if self.gap[i] > torch.tensor(1e8, device=self.device):
                self.gap[i] = self.min_gap
        self.max_gap = torch.max(self.gap)
        print('class-wise minimum distance', self.gap)
        print('min_gap', self.min_gap)
        print('max_gap', self.max_gap)

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
            
    def update_TGP(self):
        TGP = load_item(self.role, 'TGP', self.save_folder_name)
        TGP_opt = torch.optim.SGD(TGP.parameters(), lr=self.server_learning_rate)
        TGP.train()
        for e in range(self.server_epochs):
            proto_loader = DataLoader(self.uploaded_protos, self.batch_size, 
                                      drop_last=False, shuffle=True)
            for proto, y in proto_loader:
                y = torch.Tensor(y).type(torch.int64).to(self.device)

                proto_gen = TGP(list(range(self.num_classes)))

                features_square = torch.sum(torch.pow(proto, 2), 1, keepdim=True)
                centers_square = torch.sum(torch.pow(proto_gen, 2), 1, keepdim=True)
                features_into_centers = torch.matmul(proto, proto_gen.T)
                dist = features_square - 2 * features_into_centers + centers_square.T
                dist = torch.sqrt(dist)
                
                one_hot = F.one_hot(y, self.num_classes).to(self.device)
                margin = min(self.max_gap.item(), self.margin_threthold)
                dist = dist + one_hot * margin
                loss = self.CEloss(-dist, y)

                TGP_opt.zero_grad()
                loss.backward()
                TGP_opt.step()

        print(f'Server loss: {loss.item()}')
        self.uploaded_protos = []
        save_item(TGP, self.role, 'TGP', self.save_folder_name)

        TGP.eval()
        global_protos = defaultdict(list)
        for class_id in range(self.num_classes):
            global_protos[class_id] = TGP(torch.tensor(class_id, device=self.device)).detach()
        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)


def proto_cluster(protos_list):
    proto_clusters = defaultdict(list)
    for protos in protos_list:
        for k in protos.keys():
            proto_clusters[k].append(protos[k])

    for k in proto_clusters.keys():
        protos = torch.stack(proto_clusters[k])
        proto_clusters[k] = torch.mean(protos, dim=0).detach()

    return proto_clusters
            

class Trainable_Global_Prototypes(nn.Module):
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()

        self.device = device

        self.embedings = nn.Embedding(num_classes, feature_dim)
        layers = [nn.Sequential(
            nn.Linear(feature_dim, server_hidden_dim), 
            nn.ReLU()
        )]
        self.middle = nn.Sequential(*layers)
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        class_id = torch.tensor(class_id, device=self.device)

        emb = self.embedings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)

        return out