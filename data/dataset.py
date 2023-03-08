import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class DatasetFromList(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform_image = ToTensor()
        self.transform_label = lambda x: torch.tensor(x)

    def __getitem__(self, item):
        image, label = self.data[item]
        return self.transform_image(image), self.transform_label(label)

    def __len__(self):
        return len(self.data)
