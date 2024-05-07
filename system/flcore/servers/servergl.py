# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import random
import time

import torch

from flcore.clients.clientgl import clientGL
from flcore.servers.serverbase import Server
from threading import Thread
import copy

class FedGL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientGL)
        self.uploaded_heads = []
        self.aggregated_heads = [None] * self.num_clients
        self.global_head = copy.deepcopy(args.model.head)
        self.global_base = copy.deepcopy(args.model.base)
        self.heads_mixed_weights = None

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            self.send_heads()
            self.send_global_heads()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.update_heads_mixed_weight()
            self.get_aggregated_heads()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientGL)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_heads = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model.base)
                self.uploaded_heads.append(client.model.head)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def update_heads_mixed_weight(self):
        # 聚合global_head
        self.global_head = copy.deepcopy(self.uploaded_heads[0])
        for global_param in self.global_head.parameters():
            global_param.data.zero_()
        for weight, client_head in zip(self.uploaded_weights, self.uploaded_heads):
            for global_param, client_param in zip(self.global_head.parameters(), client_head.parameters()):
                global_param.data += client_param.data.clone() * weight

        # 得到heads_mixed_weight
        # 存储每个uploaded_head和global_head作差的平方
        distance_matrices = []
        # 计算作差矩阵
        global_params = list(self.global_head.parameters())  # 转换成列表以便重复使用
        for client_head in self.uploaded_heads:
            client_params = list(client_head.parameters())
            distance_matrix = [(g - l).pow(2) for g, l in zip(global_params, client_params)]
            distance_matrices.append(distance_matrix)

        # 计算混合权重
        mixed_weights=[torch.zeros_like(param) for param in global_params]
        for distance, weight in zip(distance_matrices, self.uploaded_weights):
            for mw, d in zip(mixed_weights, distance):
                mw += d * weight

        # 归一化混合权重并保持每个元素值在[0, 1]
        for i in range(len(mixed_weights)):
            mean = mixed_weights[i].mean()
            std = mixed_weights[i].std()
            standardized_weights = (mixed_weights[i] - mean) / std
            scaled_weights = 0.5 * standardized_weights + 0.5
            mixed_weights[i] = torch.clamp(scaled_weights, 0, 1)

        self.heads_mixed_weights = mixed_weights

    def get_aggregated_heads(self):
        global_params = list(self.global_head.parameters())
        for client_index, client_head in zip(self.uploaded_ids, self.uploaded_heads):
            client_params = list(client_head.parameters())
            aggregated_head = copy.deepcopy(self.global_head)
            aggregated_params = list(aggregated_head.parameters())
            with torch.no_grad():
                for gp, ap, mw, lp in zip(global_params, aggregated_params, self.heads_mixed_weights, client_params):
                    ap.data = gp.data + (lp.data - gp.data) * mw
            self.aggregated_heads[client_index] = aggregated_head

            # print(f"aggregated_head: {aggregated_head}")
            # print(f"aggregated_params: {aggregated_params}")
            # print(f"heads_mixed_weights: {self.heads_mixed_weights}")
            # time.sleep(100)

    def send_global_heads(self):
        for client in self.clients:
            client.set_global_head_parameters(self.global_head)
    def send_heads(self):
        for client_index, client in enumerate(self.clients):
            if client_index in self.uploaded_ids and self.aggregated_heads[client_index] is not None:
                client.set_head_parameters(self.aggregated_heads[client_index])