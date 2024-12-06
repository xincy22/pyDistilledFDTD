import pytest
import torch
from src.data.processors.pca import PCATransformer
from src.data.base import BaseDataset

class TestPCATransformer:
    @pytest.fixture
    def sample_dataset(self):
        """创建示例数据集"""
        data = torch.randn(100, 20)  # 100个样本，20维
        labels = torch.randint(0, 10, (100,))
        return BaseDataset(data, labels)

    @pytest.fixture
    def pca_transformer(self):
        """创建PCA转换器"""
        return PCATransformer(n_components=10)

    def test_init(self, pca_transformer):
        """测试初始化"""
        assert pca_transformer.n_components == 10
        assert pca_transformer.pca is None
        assert pca_transformer.device == "cpu"

    def test_fit_transform(self, pca_transformer, sample_dataset):
        """测试拟合和转换"""
        transformed = pca_transformer.fit_transform(sample_dataset)
        
        # 验证输出类型
        assert isinstance(transformed, BaseDataset)
        
        # 验证维度降低
        assert transformed.data.shape[1] == 10
        
        # 验证标签保持不变
        assert torch.equal(transformed.labels, sample_dataset.labels)

    def test_transform(self, pca_transformer, sample_dataset):
        """测试转换"""
        # 先进行fit_transform
        pca_transformer.fit_transform(sample_dataset)
        
        # 然后测试transform
        transformed = pca_transformer.transform(sample_dataset)
        assert isinstance(transformed, BaseDataset)
        assert transformed.data.shape[1] == 10