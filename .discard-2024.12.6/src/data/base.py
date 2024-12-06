import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor, device: str = "cpu"):
        self.data = data.to(device)
        self.labels = labels.to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

