import time
import copy
import  torch
from flcore.clients.clientpaw import clientPAW
from flcore.servers.serverbase import Server
from threading import Thread


class FedPAW(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientPAW)

        self.personalized_models = [None] * self.num_clients
        self.mixed_weights = None

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.rounds =args.rounds
        self.layer_idx = args.layer_idx


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models(i)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()


            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            if i >= self.rounds:
                # Update mixed weights and generate personalized models
                self.update_mixed_weight()
                self.generate_personalized_models()

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
        self.save_global_model()
        self.save_personalized_models()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientPAW)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def send_models(self, round):
        assert (len(self.clients) > 0)

        for client_index, client in enumerate(self.clients):
            if round > self.rounds:
                if client_index in self.uploaded_ids:
                    client.set_parameters(self.personalized_models[client_index])
            else:
                client.set_parameters(self.global_model)


    def update_mixed_weight(self):
        # 获取全局模型的参数
        params_g = list(self.global_model.parameters())

        # 存储每个本地模型和全局模型作差的平方
        distance_matrices = []

        # 计算作差矩阵
        for client_model in self.uploaded_models:
            params_l = list(client_model.parameters())
            distance_matrix = [
                (param_g - param_l).pow(2)
                for param_g, param_l in zip(params_g, params_l)]
            distance_matrices.append(distance_matrix)

        # 计算混合权重
        mixed_weights = [
            torch.zeros_like(param) for param in params_g
        ]

        for distance_matrix, weight in zip(distance_matrices, self.uploaded_weights):
            for i, distance in enumerate(distance_matrix):
                mixed_weights[i] += distance * weight

        # 归一化混合权重并保持每个元素值在[0, 1]
        for i, weight in enumerate(mixed_weights):
            min_val = weight.min()
            max_val = weight.max()
            # 避免除以0的情况，当所有值相等时，可以将该Tensor设为全0或根据需要进行处理
            if max_val - min_val > 0:
                mixed_weights[i] = (weight - min_val) / (max_val - min_val)
            else:
                # 可以选择将其设为全0或其他处理方式
                mixed_weights[i] = torch.zeros_like(weight)



        self.mixed_weights = mixed_weights

        # # 打印mixed_weights形状确认
        # print("Mixed Weights Shape Check:")
        # for weight in mixed_weights:
        #     print(weight.shape)


    def generate_personalized_models(self):
        # Determine the index from which to start applying mixed weights
        start_idx = -self.layer_idx

        for index in range(len(self.personalized_models)):
            self.personalized_models[index] = copy.deepcopy(self.global_model)

            # Use the mixed weights to generate personalized models for each client
        for client_index, client_model in zip(self.uploaded_ids, self.uploaded_models):
            # Start with the global model parameters
            global_params = list(self.global_model.parameters())

            # Get the local model parameters
            local_params = list(client_model.parameters())

            # Deep copy the global model as the starting point for the personalized model
            personalized_model = copy.deepcopy(self.global_model)
            personalized_params = list(personalized_model.parameters())

            # Apply mixed weights to combine global and local models for higher layers
            with torch.no_grad():
                for idx in range(start_idx, len(global_params)):
                    global_param = global_params[idx]
                    local_param = local_params[idx]
                    mixed_weight = self.mixed_weights[idx]
                    personalized_param = personalized_params[idx]

                    # Compute the personalized parameters for higher layers
                    personalized_param.data = global_param.data + (local_param.data - global_param.data) * mixed_weight

            self.personalized_models[client_index] = personalized_model


    def save_personalized_models(self):
        for index, p_model in enumerate(self.personalized_models):
            if p_model is not None:  # Check if the personalized model exists
                self.save_model(p_model, f"personalized_model_{index}")