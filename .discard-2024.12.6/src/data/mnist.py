import os
from typing import Tuple

import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from config.data_config import RAW_DATA_DIR
from .base import BaseDataset


class MNISTDataset:
    """MNIST数据集加载器"""

    def __init__(self, device: str = "cpu"):
        """初始化MNIST数据集加载器"""
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        self.device = device

    def load_dataset(self) -> Tuple[BaseDataset, BaseDataset]:
        """加载MNIST数据集并返回训练集和测试集"""
        train_dataset = datasets.MNIST(
            root=str(RAW_DATA_DIR),
            train=True, 
            download=True
        )
        test_dataset = datasets.MNIST(
            root=str(RAW_DATA_DIR),
            train=False,
            download=True
        )

        train_data = train_dataset.data.float() / 255.0
        train_data = train_data.view(train_data.size(0), -1)
        test_data = test_dataset.data.float() / 255.0
        test_data = test_data.view(test_data.size(0), -1)

        train_set = BaseDataset(train_data, train_dataset.targets, device=self.device)
        test_set = BaseDataset(test_data, test_dataset.targets, device=self.device)

        return train_set, test_set
    
    def get_dataloader(self, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
        """获取MNIST数据集的DataLoader"""
        train_set, test_set = self.load_dataset()
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

