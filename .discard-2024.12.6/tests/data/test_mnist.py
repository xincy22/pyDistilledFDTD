import pytest
import torch
from src.data.mnist import MNISTDataset
from src.data.base import BaseDataset
from torch.utils.data import DataLoader

class TestMNISTDataset:
    @pytest.fixture
    def mnist_dataset(self):
        """创建MNIST数据集实例"""
        return MNISTDataset(device="cpu")

    def test_init(self, mnist_dataset):
        """测试初始化"""
        assert mnist_dataset.device == "cpu"

    def test_load_dataset(self, mnist_dataset):
        """测试数据集加载"""
        train_set, test_set = mnist_dataset.load_dataset()
        
        # 验证返回类型
        assert isinstance(train_set, BaseDataset)
        assert isinstance(test_set, BaseDataset)
        
        # 验证数据维度
        assert train_set.data.shape[1] == 784  # 28*28=784
        assert test_set.data.shape[1] == 784
        
        # 验证数据范围
        assert torch.all(train_set.data >= 0)
        assert torch.all(train_set.data <= 1)
        
        # 验证标签范围
        assert torch.all(train_set.labels >= 0)
        assert torch.all(train_set.labels <= 9)

    def test_get_dataloader(self, mnist_dataset):
        """测试数据加载器获取"""
        train_loader, test_loader = mnist_dataset.get_dataloader()

        # 验证返回类型
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

        # 验证batch大小
        assert train_loader.batch_size == 64
        assert test_loader.batch_size == 64

        # 验证数据集大小
        assert len(train_loader.dataset) > 0
        assert len(test_loader.dataset) > 0

        # 验证shuffle设置（通过检查多次迭代是否返回相同顺序）
        if len(train_loader) > 1:
            first_batch_1 = next(iter(train_loader))
            train_loader_2, _ = mnist_dataset.get_dataloader()
            first_batch_2 = next(iter(train_loader_2))
            # 由于shuffle=True，两次获取的第一个batch应该不同
            assert not torch.equal(first_batch_1[0], first_batch_2[0])

        # 对于测试集，由于shuffle=False，顺序应该相同
        if len(test_loader) > 1:
            first_batch_1 = next(iter(test_loader))
            _, test_loader_2 = mnist_dataset.get_dataloader()
            first_batch_2 = next(iter(test_loader_2))
            # 由于shuffle=False，两次获取的第一个batch应该相同
            assert torch.equal(first_batch_1[0], first_batch_2[0])