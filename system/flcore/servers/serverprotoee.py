import time
from collections import defaultdict

from flcore.clients.clientprotoee import clientProto
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item


class FedProto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientProto)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes


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
        uploaded_proto_maps = []
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
                    uploaded_proto_maps.append(class_map)
                    per_exit_counts[exit_id] += len(class_map)
            else:
                uploaded_proto_maps.append(protos_by_exit)

        if not uploaded_proto_maps:
            return

        global_protos = proto_aggregation(uploaded_proto_maps)
        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)

        if per_exit_counts:
            total = sum(per_exit_counts.values())
            exit_str = ", ".join([f"{eid}:{cnt}" for eid, cnt in sorted(per_exit_counts.items())])
            print(f"[FedProtoEE] proto_usage total={total} per_exit=[{exit_str}]")

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
    

def proto_aggregation(local_protos_list):
    agg_protos = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos[label].append(local_protos[label])

    for [label, proto_list] in agg_protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos[label] = proto / len(proto_list)
        else:
            agg_protos[label] = proto_list[0].data

    return agg_protos