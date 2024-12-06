import pytest
import torch
from src.data.pipeline import DataPipeline

class TestDataPipeline:
    def setup_method(self):
        self.pipeline = DataPipeline()
        
    def test_init_params(self):
        """测试初始化参数"""
        pipeline = DataPipeline(
            n_components=10,
            batch_size=32,
            use_core_set=True
        )
        assert pipeline.n_components == 10
        assert pipeline.batch_size == 32
        assert pipeline.use_core_set is True
        
    def test_invalid_params(self):
        """测试无效参数"""
        with pytest.raises(ValueError):
            DataPipeline(n_components=0)
        with pytest.raises(ValueError):
            DataPipeline(batch_size=0)
        with pytest.raises(ValueError):
            DataPipeline(eta=0)
            
    def test_data_dimensions(self):
        """测试数据维度"""
        loaders = self.pipeline.get_dataloader()
        batch = next(iter(loaders["train"]))
        # 检查 PCA 降维后的维度
        assert batch[0].shape[1] == self.pipeline.n_components
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
    def test_gpu_support(self):
        """测试 GPU 支持"""
        pipeline = DataPipeline(device="cuda")
        loaders = pipeline.get_dataloader()
        batch = next(iter(loaders["train"]))
        assert batch[0].device.type == "cuda"
        
    def test_core_set_effect(self):
        """测试核心集选择效果"""
        pipeline = DataPipeline(use_core_set=True, eta=0.5)
        loaders = pipeline.get_dataloader()
        # 检查核心集大小是否正确
        n_samples = len(loaders["train"].dataset)
        expected_size = int(60000 * 0.5)  # MNIST 训练集大小 * eta
        assert n_samples == expected_size