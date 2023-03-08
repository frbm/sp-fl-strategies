from fedavg.server import *
from fedprox.client import *


class FedProxServer(FedAvgServer):
    def __init__(self, num_clients, subset_size, model, loss, mu, train_loaders,
                 test_loader, local_epochs, local_lr, max_samples,
                 min_samples=None, p_strategy='uniform'):
        if min_samples is None:
            min_samples = [None] * num_clients

        super().__init__(num_clients, subset_size, model, loss, train_loaders,
                         test_loader, local_epochs, local_lr, max_samples,
                         min_samples, p_strategy)

        self.clients = [
            FedProxClient(self.device,
                          loss,
                          local_epochs,
                          local_lr,
                          mu,
                          train_loaders[k],
                          max_samples[k],
                          min_samples[k]
                          ) for k in range(num_clients)
        ]

        self.mu = mu
