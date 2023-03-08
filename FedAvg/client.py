import numpy as np
from torch import optim
from copy import deepcopy


class FedAvgClient:
    def __init__(self, device, loss, epochs, lr, train_loader, max_samples,
                 min_samples=None):
        self.device = device
        self.loss_function = loss
        self.epochs = epochs
        self.lr = lr
        self.train_loader = train_loader
        self.max_samples = max_samples
        self.min_samples = min_samples

    def update(self, model):
        if self.min_samples is not None:
            num_samples = np.random.randint(self.min_samples, self.max_samples)
        else:
            num_samples = self.max_samples

        model = deepcopy(model)
        model.to(self.device)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            for x, label in self.train_loader:
                x, label = x.to(self.device), label.to(self.device)
                optimizer.zero_grad()
                pred = model(x)
                loss = self.loss_function(pred, label)
                loss.backward()
                optimizer.step()

        weights = list(model.parameters())
        weights = [w.detach().cpu() for w in weights]
        del model  # save memory
        return weights, num_samples, loss.item()
