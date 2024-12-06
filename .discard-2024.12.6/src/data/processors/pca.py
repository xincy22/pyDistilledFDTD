import torch
from sklearn.decomposition import PCA
from ..base import BaseDataset
from ...utils.logger import LoggerManager

logger = LoggerManager.get_logger(__name__)

class PCATransformer:
    """PCA降维转换器"""

    def __init__(self, n_components: int = 10, device: str = "cpu"):
        """初始化PCA转换器"""
        self.n_components = n_components
        self.pca = None  # PCA模型，需要在fit_transform时初始化
        self.device = device
        
    def fit_transform(self, dataset: BaseDataset) -> BaseDataset:
        """训练PCA模型并转换数据"""
        # 将数据转换为numpy格式用于PCA计算
        data_np = dataset.data.detach().cpu().numpy()
        
        # 训练并应用PCA转换
        self.pca = PCA(n_components=self.n_components)
        data_pca = self.pca.fit_transform(data_np)
        
        # 标准化处理：减去均值并除以标准差
        data_pca = (data_pca - data_pca.mean(axis=0)) / data_pca.std(axis=0)
        
        # 将结果转回tensor格式并保持设备和数据类型一致
        data_tensor = torch.tensor(data_pca, device=dataset.data.device, dtype=dataset.data.dtype)
        returns = BaseDataset(data_tensor, dataset.labels, device=self.device)
        
        logger.info(f"PCA transform finished, dataset shape: {returns.data.shape}")
        return returns

    def transform(self, dataset: BaseDataset) -> BaseDataset:
        """使用已训练的PCA模型转换数据"""
        # 检查PCA模型是否已训练
        if self.pca is None:
            logger.error("PCA model is not fitted, please call fit_transform first")
            raise RuntimeError("PCA model is not fitted, please call fit_transform first")
        
        # 将数据转换为numpy格式
        data_np = dataset.data.detach().cpu().numpy()
        
        # 应用PCA转换
        data_pca = self.pca.transform(data_np)
        
        # 标准化处理
        data_pca = (data_pca - data_pca.mean(axis=0)) / data_pca.std(axis=0)
        
        # 转回tensor格式
        data_tensor = torch.tensor(data_pca, device=dataset.data.device, dtype=dataset.data.dtype)
        returns = BaseDataset(data_tensor, dataset.labels, device=self.device)
        
        logger.info(f"PCA transform finished, dataset shape: {returns.data.shape}")
        return returns
