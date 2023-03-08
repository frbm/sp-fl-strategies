import numpy as np
from data.dataset import *


def iid_split(dataset, num_clients):
    datasets = []
    data_points = np.arange(len(dataset))
    np.random.shuffle(data_points)

    splits = np.array_split(data_points, num_clients)

    for k in range(num_clients):
        datasets.append(
            DatasetFromList(
                [dataset[data_points[splits[k]][i]] for i in range(len(
                    splits[k]))]
            )
        )
    return datasets


def dirichlet_split(dataset, num_clients, alpha=None):
    if alpha is None:
        alpha = np.random.rand(num_clients)
    datasets = []
    data_points = np.arange(len(dataset))
    np.random.shuffle(data_points)

    proportions = np.random.dirichlet(alpha)
    splits = np.cumsum([0] + [int(p * len(dataset)) for p in proportions])
    for k in range(1, len(splits)):  # avoid clients with an empty dataset
        if splits[k] == splits[k - 1]:
            splits[k] += 1

    for k in range(num_clients):
        indices = np.arange(splits[k], splits[k + 1])
        datasets.append(
            DatasetFromList([dataset[data_points[i]] for i in indices])
        )
    return datasets


def class_split(dataset, num_clients, clients_classes=None, verbose=True):
    """
    :param dataset: torch.utils.data.Dataset object, must have a classes
    attribute.
    :param num_clients:
    :param clients_classes:
    :param verbose:
    """
    datasets = [[] for _ in range(num_clients)]
    data_points = np.arange(len(dataset))
    np.random.shuffle(data_points)

    classes = np.arange(len(dataset.classes))

    data_points_classes = [[] for _ in range(len(classes))]
    for x in data_points:
        data_points_classes[dataset[x][1]].append(x)

    if clients_classes is None:
        num_classes = np.random.randint(1, len(classes))
        clients_classes = [
            np.random.choice(classes, num_classes, replace=False) for _ in
            range(num_clients)
        ]

    for index in classes:
        clients = [client for client in range(num_clients) if index in
                   clients_classes[client]]
        if len(clients) == 0:
            clients.append(np.random.randint(num_clients))
        if verbose:
            print(f"Clients for class {index}: {clients}")
        data_points = np.array(data_points_classes[index])
        splits = np.array_split(data_points, len(clients))
        for i in range(len(clients)):
            datasets[clients[i]] += list(splits[i])

    datasets = [DatasetFromList(dataset) for dataset in datasets]
    return datasets
