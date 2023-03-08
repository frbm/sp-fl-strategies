import torch
from fedavg.client import *


class FedAvgServer:
    def __init__(self, num_clients, subset_size, model, loss,
                 train_loaders, test_loader, local_epochs, local_lr,
                 max_samples, min_samples=None, p_strategy='uniform'):

        if min_samples is None:
            min_samples = [None] * num_clients

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.loss_function = loss

        self.clients_ids = np.arange(num_clients)
        self.clients = [
            FedAvgClient(self.device,
                         loss,
                         local_epochs,
                         local_lr,
                         train_loaders[k],
                         max_samples[k],
                         min_samples[k]
                         ) for k in range(num_clients)
        ]
        self.num_clients = num_clients

        self.subset = None
        self.subset_size = subset_size
        if p_strategy == 'uniform':
            self.choice_p = np.ones(num_clients) / num_clients
        else:
            self.choice_p = np.array([len(loader) for loader in
                                      train_loaders], dtype=np.float)
            self.choice_p /= self.choice_p.sum()

        self.test_loader = test_loader

    def make_subset(self):
        self.subset = np.random.choice(
            self.clients_ids,
            self.subset_size,
            replace=False,
            p=self.choice_p
        )

    def zero_model(self):
        model = deepcopy(self.model)
        for layer_weights in model.parameters():
            layer_weights.data.sub_(layer_weights.data)
        return model

    def train(self, total_steps, trace_loss=True, verbose=True):
        local_losses = np.zeros((total_steps, self.num_clients))
        global_losses = np.zeros(total_steps)
        global_accuracy = np.zeros(total_steps)

        for t in range(total_steps):
            # initialization
            self.make_subset()
            model = self.zero_model()
            clients_weights = []
            clients_samples = []

            # clients updates
            for k in self.subset:
                client_weights, num_samples, local_loss = self.clients[
                    k].update(self.model)
                clients_weights.append(client_weights)
                clients_samples.append(num_samples)
                local_losses[t][k] = local_loss
            total_samples = np.sum(clients_samples)

            # aggregation
            for k, client_weights in enumerate(clients_weights):
                for i, layer_weights in enumerate(model.parameters()):
                    contribution = client_weights[i].data * clients_samples[
                        k] / total_samples
                    layer_weights.data.add_(contribution)
            self.model = model

            if trace_loss:
                global_losses[t], global_accuracy[t] = self.eval_model()
                if verbose:
                    print(f"Step {t + 1} of {total_steps}.")
                    print(f"Global loss: {global_losses[t]}")
                    print(f"Global accuracy: {100 * global_accuracy[t]}%.")
                    print()

        if trace_loss:
            return global_losses, local_losses

    @torch.no_grad()
    def eval_model(self):
        model = deepcopy(self.model)
        model.to(self.device)
        model.eval()
        loss = 0.
        accuracy = 0.
        for x, label in self.test_loader:
            x, label = x.to(self.device), label.to(self.device)
            pred = model(x)
            loss += self.loss_function(pred, label).item()
            accuracy += float(label == torch.argmax(pred))
        loss /= len(self.test_loader)
        accuracy /= len(self.test_loader)
        return loss, accuracy
