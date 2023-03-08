import torch
from scaffold.client import *


class ScaffoldServer:
    def __init__(self, num_clients, subset_size, model, loss,
                 train_loaders, test_loader, local_epochs, local_lr, global_lr,
                 max_samples, min_samples=None, c_init='zeros'):
        """

        :param num_clients:
        :param subset_size:
        :param model:
        :param loss:
        :param train_loaders:
        :param test_loader:
        :param local_epochs:
        :param local_lr:
        :param global_lr:
        :param max_samples:
        :param min_samples:
        :param c_init: 'zeros' or 'uniform' or 'gaussian' for standard Gaussian.
        """
        if min_samples is None:
            min_samples = [None] * num_clients

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.zero_model()
        self.loss_function = loss
        self.global_lr = global_lr

        if c_init == 'zeros':
            self.c = [torch.zeros_like(p) for p in self.model.parameters()]
        elif c_init == 'uniform':
            self.c = [torch.rand_like(p) for p in self.model.parameters()]
        elif c_init == 'gaussian':
            self.c = [torch.randn_like(p) for p in self.model.parameters()]
        else:
            raise NotImplementedError
        local_c_init = [
            [p / num_clients for p in deepcopy(self.c)]
            for _ in range(num_clients)
        ]

        self.clients_ids = np.arange(num_clients)
        self.clients = [
            ScaffoldClient(self.device,
                           loss,
                           local_epochs,
                           local_lr,
                           train_loaders[k],
                           max_samples[k],
                           min_samples[k],
                           local_c_init[k],
                           )
            for k in range(num_clients)
        ]
        self.num_clients = num_clients

        self.subset = None
        self.subset_size = subset_size
        self.choice_p = np.array([len(loader) for loader in train_loaders],
                                 dtype=np.float)
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
        for layer_weights in self.model.parameters():
            layer_weights.data.sub_(layer_weights.data)

    def train(self, total_steps, trace_loss=True, verbose=True):
        local_losses = np.zeros((total_steps, self.num_clients))
        global_losses = np.zeros(total_steps)
        global_accuracy = np.zeros(total_steps)

        for t in range(total_steps):
            # initialization
            self.make_subset()
            clients_delta_weights = []
            clients_delta_cov = []

            # clients updates
            for k in self.subset:
                delta_weights, delta_cov, local_loss = self.clients[
                    k].update(self.model, self.c)
                clients_delta_weights.append(delta_weights)
                clients_delta_cov.append(delta_cov)
                local_losses[t][k] = local_loss

            # aggregation
            for client_delta_weights in clients_delta_weights:
                for i, layer_weights in enumerate(self.model.parameters()):
                    contrib = self.global_lr * client_delta_weights[i].data
                    layer_weights.data.add_(contrib)
                    
            for client_delta_cov in clients_delta_cov:
                for i, layer_weights in enumerate(self.model.parameters()):
                    contrib = self.subset_size / self.num_clients * \
                              client_delta_cov[i].data
                    layer_weights.data.add_(contrib)

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
