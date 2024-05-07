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
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
import copy


class clientGL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.plocal_epochs = args.plocal_epochs
        self.aggregated = args.aggregated
        self.mu_head = args.mu_head
        self.mu_base = args.mu_base

        self.global_head_params = copy.deepcopy(list(self.model.head.parameters()))
        # 该模型具有全局base和本地头，与fedrep算法一样
        self.fedrep_model = copy.deepcopy(args.model)

        # 使用自定义base优化器
        self.optimizer_base =PerturbedGradientDescent(
            self.model.base.parameters(), lr=self.learning_rate, mu=self.mu_head)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_base,
            gamma=args.learning_rate_decay_gamma
        )

        self.optimizer_rep_base = torch.optim.SGD(self.fedrep_model.base.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_rep_base,
            gamma=args.learning_rate_decay_gamma
        )

        # 使用自定义head优化器
        self.optimizer_per = PerturbedGradientDescent(
            self.model.head.parameters(), lr=self.learning_rate, mu=self.mu_head)

        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per,
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self.fedrep_model.to(self.device)
        self.model.train()
        self.fedrep_model.train()
        # 第一步，更新本地头
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
                self.optimizer_per.step(self.global_head_params, self.device)
        #更新fedrep_model的本地头
        self.fedrep_model.head = copy.deepcopy(self.model.head)

        # 第二步，固定本地头，更新fedrep_model中的base
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # for param in self.fedrep_model.base.parameters():
        #     param.requires_grad = True
        # for param in self.fedrep_model.head.parameters():
        #     param.requires_grad = False
        #
        # for epoch in range(max_local_epochs):
        #     for i, (x, y) in enumerate(trainloader):
        #         if type(x) == type([]):
        #             x[0] = x[0].to(self.device)
        #         else:
        #             x = x.to(self.device)
        #         y = y.to(self.device)
        #         if self.train_slow:
        #             time.sleep(0.1 * np.abs(np.random.rand()))
        #         output = self.fedrep_model(x)
        #         loss = self.loss(output, y)
        #         self.optimizer_rep_base.zero_grad()
        #         loss.backward()
        #         self.optimizer_rep_base.step()

        # 第三步，继续固定本地头，固定fedrep_model，增加fedrep_model和self.model的中间表示的l2范数正则项，更新self.model的base
        self.fedrep_model.eval()
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
                feat1 = self.model.base(x)
                feat2 = self.fedrep_model.base(x)
                output = self.model.head(feat1)
                regularization_loss = 0.5 * self.mu_base * torch.norm(feat1 - feat2, p=2)
                loss = self.loss(output, y) + regularization_loss
                self.optimizer_base.zero_grad()
                loss.backward()
                self.optimizer_base.step(self.fedrep_model.base.parameters(), self.device)

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_per.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, base):
        # 更新全局base
        for new_param, old_param in zip(base.parameters(), self.fedrep_model.base.parameters()):
            old_param.data = new_param.data.clone()

    def set_global_head_parameters(self, head):
        # 更新全局头
        for new_param, old_param in zip(head.parameters(), self.global_head_params):
            old_param.data = new_param.data.clone()

    def set_head_parameters(self, head):
        # 得到聚合后的头
        if self.aggregated:
            for new_param, old_param in zip(head.parameters(), self.model.head.parameters()):
                old_param.data = new_param.data.clone()

