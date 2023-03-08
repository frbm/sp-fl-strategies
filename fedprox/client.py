import numpy as np
from torch import optim
from copy import deepcopy
from fedavg.client import *


class FedProxClient(FedAvgClient):
    def __init__(self, device, loss, epochs, lr, mu, train_loader, max_samples,
                 min_samples=None):
        super().__init__(device, loss, epochs, lr, train_loader, max_samples,
                         min_samples)
        self.mu = mu

    def update(self, global_model):
        if self.min_samples is not None:
            num_samples = np.random.randint(self.min_samples, self.max_samples)
        else:
            num_samples = self.max_samples

        model = deepcopy(global_model)
        model.to(self.device)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        loss = 0.

        for epoch in range(self.epochs):
            for x, label in self.train_loader:
                x, label = x.to(self.device), label.to(self.device)
                optimizer.zero_grad()
                pred = model(x)
                loss = self.loss_function(pred, label)
                loss.backward()
                for w, w_g in zip(model.parameters(),
                                  global_model.parameters()):
                    w_g = w_g.to(self.device)
                    w.grad.data += self.mu * (w_g.data - w.data)
                optimizer.step()

        weights = list(model.parameters())
        weights = [w.detach().cpu() for w in weights]
        del model  # save memory
        return weights, num_samples, loss.item()
