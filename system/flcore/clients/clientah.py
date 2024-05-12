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

import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
import copy

class clientAH(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.optimizer_per = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per, 
            gamma=args.learning_rate_decay_gamma
        )

        self.plocal_epochs = args.plocal_epochs
        self.global_model = copy.deepcopy(args.model)
        self.head_weights = [torch.ones_like(param.data).to(self.device)
                             for param in list(self.global_model.head.parameters())]
        self.eta = 1.0  # Weight learning rate. Default: 1.0

    def train(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        # aggregate head
        # obtain the references of the head parameters
        params_g = list(self.global_model.head.parameters())
        params = list(self.model.head.parameters())

        # temp local model only for head weights learning
        model_t = copy.deepcopy(self.model)
        params_th = list(model_t.head.parameters())
        params_tb = list(model_t.base.parameters())

        # frozen base to reduce computational cost in Pytorch
        for param in params_tb:
                param.requires_grad = False

        # used to obtain the gradient of model, no need to use optimizer.step(), so lr=0
        optimizer_t = torch.optim.SGD(model_t.parameters(), lr=0)

        for param_t, param, param_g, weight in zip(params_th, params, params_g, self.head_weights):
            param_t.data = param + (param_g - param) * weight

        for epoch in range(self.plocal_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                optimizer_t.zero_grad()
                output = model_t(x)
                loss_value = self.loss(output, y)
                loss_value.backward()

                # update head weights in this batch
                for param_t, param, param_g, weight in zip(params_th, params,
                                                        params_g, self.head_weights):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_th, params,
                                                        params_g, self.head_weights):
                    param_t.data = param + (param_g - param) * weight

        # obtain initialized aggregated head
        for param, param_t in zip(params, params_th):
            param.data = param_t.data.clone()


        # train head
        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

        for epoch in range(self.plocal_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer_per.zero_grad()
                loss.backward()
                self.optimizer_per.step()
                
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # train base
        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_per.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        
            
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.global_model.parameters()):
            old_param.data = new_param.data.clone()

        for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

