import pytest
import torch
from src.data.processors.core_select import CoreSelector
from src.data.base import BaseDataset

class TestCoreSelector:
    @pytest.fixture
    def sample_dataset(self):
        """创建示例数据集"""
        data = torch.randn(1000, 10)  # 1000个样本，10维
        labels = torch.randint(0, 10, (1000,))
        return BaseDataset(data, labels)

    @pytest.fixture
    def core_selector(self):
        """创建核心集选择器"""
        return CoreSelector(eta=0.1, method="greedy")

    def test_init(self, core_selector):
        """测试初始化"""
        assert core_selector.eta == 0.1
        assert core_selector.method == "greedy"
        assert core_selector.device == "cpu"

    def test_fit_transform_greedy(self, core_selector, sample_dataset):
        """测试贪心算法选择"""
        transformed = core_selector.fit_transform(sample_dataset)
        
        # 验证核心集大小
        expected_size = int(len(sample_dataset) * core_selector.eta)
        assert len(transformed) == expected_size

    def test_fit_transform_kmeans(self, sample_dataset):
        """测试K-means选择"""
        kmeans_selector = CoreSelector(eta=0.1, method="kmeans")
        transformed = kmeans_selector.fit_transform(sample_dataset)
        
        expected_size = int(len(sample_dataset) * kmeans_selector.eta)
        assert len(transformed) == expected_size

    def test_invalid_method(self):
        """测试无效的方法"""
        with pytest.raises(ValueError):
            CoreSelector(method="invalid")