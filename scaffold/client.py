import numpy as np
import torch
from torch import optim
from copy import deepcopy


class ScaffoldClient:
    def __init__(self, device, loss, epochs, local_lr, train_loader,
                 max_samples, min_samples=None, local_c=None):
        self.device = device
        self.loss_function = loss
        self.epochs = epochs
        self.lr = local_lr
        self.c = local_c
        self.delta_c = None
        self.train_loader = train_loader
        self.max_samples = max_samples
        self.min_samples = min_samples

    def update(self, global_model, global_c):
        if self.min_samples is not None:
            num_samples = np.random.randint(self.min_samples, self.max_samples)
        else:
            num_samples = self.max_samples

        if self.delta_c is None:
            if self.c is None:
                self.delta_c = [torch.zeros_like(p) for p in global_c]
            else:
                self.delta_c = [c_g - c_i for (c_g, c_i) in
                                zip(global_c, self.c)]

        local_model = deepcopy(global_model)
        local_model.to(self.device)
        local_model.train()

        optimizer = optim.Adam(local_model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            for x, label in self.train_loader:
                x, label = x.to(self.device), label.to(self.device)
                optimizer.zero_grad()
                pred = local_model(x)
                loss = self.loss_function(pred, label)
                loss.backward()
                for p, c_diff in zip(local_model.parameters(), self.delta_c):
                    p.grad += c_diff.data
                optimizer.step()

        local_model = list(local_model.parameters())
        local_model = [w.detach().cpu() for w in local_model]
        delta_weights = [p_l - p_g for (p_l, p_g) in zip(
            local_model, global_model.parameters())]
        weighted_lr = 1 / (num_samples * self.lr)
        self.c = [weighted_lr * w_d - c_d for (w_d, c_d) in zip(delta_weights,
                                                                self.delta_c)]
        self.delta_c = [c_g - c_i for (c_g, c_i) in zip(global_c, self.c)]
        return delta_weights, self.delta_c, loss.item()

    def train(self):
        pass
