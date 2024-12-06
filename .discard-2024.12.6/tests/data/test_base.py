import pytest
import torch
from src.data.base import BaseDataset

class TestBaseDataset:
    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        data = torch.randn(100, 10)  # 100个样本，每个10维
        labels = torch.randint(0, 10, (100,))  # 100个标签，范围0-9
        return data, labels

    def test_init(self, sample_data):
        """测试初始化"""
        data, labels = sample_data
        dataset = BaseDataset(data, labels)
        
        assert torch.equal(dataset.data, data)
        assert torch.equal(dataset.labels, labels)
        assert dataset.data.device == torch.device('cpu')
        assert dataset.labels.device == torch.device('cpu')

    def test_len(self, sample_data):
        """测试长度计算"""
        data, labels = sample_data
        dataset = BaseDataset(data, labels)
        assert len(dataset) == 100

    def test_getitem(self, sample_data):
        """测试索引访问"""
        data, labels = sample_data
        dataset = BaseDataset(data, labels)
        
        x, y = dataset[0]
        assert torch.equal(x, data[0])
        assert torch.equal(y, labels[0])

    def test_device_placement(self, sample_data):
        """测试设备放置"""
        data, labels = sample_data
        
        # 测试CPU设备
        dataset = BaseDataset(data, labels, device="cpu")
        assert dataset.data.device == torch.device('cpu')
        assert dataset.labels.device == torch.device('cpu')
        
        # 测试CUDA设备
        if torch.cuda.is_available():
            dataset = BaseDataset(data, labels, device="cuda")
            assert dataset.data.device.type == 'cuda'  # 只检查设备类型
            assert dataset.labels.device.type == 'cuda'