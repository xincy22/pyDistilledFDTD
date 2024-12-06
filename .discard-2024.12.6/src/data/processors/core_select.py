import numpy as np
from sklearn.cluster import KMeans
from ...utils.logger import LoggerManager
from ..base import BaseDataset

logger = LoggerManager.get_logger(__name__)

class CoreSelector:
    """核心集选择器，用于选择数据集的代表性样本"""

    def __init__(self, eta: float = 0.1, method: str = "kmeans", device: str = "cpu"):
        """初始化核心集选择器"""
        if eta <= 0 or eta >= 1:
            raise ValueError("eta must be between 0 and 1")
        if method not in ["greedy", "kmeans"]:
            raise ValueError("method must be either 'greedy' or 'kmeans'")
        self.eta = eta  # 核心集占原始数据集的比例
        self.method = method  # 选择方法：kmeans或greedy
        self.device = device

    def fit_transform(self, dataset: BaseDataset) -> BaseDataset:
        """选择核心集并返回"""
        data_np = dataset.data.detach().cpu().numpy()
        if self.method == "greedy":
            indices = self._select_greedy(data_np)
        elif self.method == "kmeans":
            indices = self._select_kmeans(data_np)
        else:
            logger.error(f"Invalid method: {self.method}")
            raise ValueError(f"Invalid method: {self.method}")
        returns = BaseDataset(dataset.data[indices], dataset.labels[indices], device=self.device)
        logger.info(f"Core selection finished, dataset shape: {returns.data.shape}")
        return returns
    
    def transform(self, dataset: BaseDataset) -> BaseDataset:
        """测试集不需要选择核心集，直接返回"""
        return dataset
    
    def _select_greedy(self, data_np: np.ndarray) -> np.ndarray:
        """贪心算法选择核心集"""
        N = data_np.shape[0]
        core_size = int(N * self.eta)
        core_indices = []

        # 随机选择第一个点
        first_idx = np.random.choice(N)
        core_indices.append(first_idx)
        min_dist = np.linalg.norm(data_np - data_np[first_idx], axis=1)

        # 迭代选择剩余的点
        for _ in range(1, core_size):
            next_idx = np.argmax(min_dist)
            core_indices.append(next_idx)
            dist = np.linalg.norm(data_np - data_np[next_idx], axis=1)
            min_dist = np.minimum(min_dist, dist)

        logger.info(f"Greedy core selection finished, selected {len(core_indices)} cores")
        return core_indices
    
    def _select_kmeans(self, data_np: np.ndarray) -> np.ndarray:
        """K-means聚类选择核心集"""
        N = data_np.shape[0]
        core_size = int(N * self.eta)

        # 执行K-means聚类
        kmeans = KMeans(n_clusters=core_size, random_state=42, n_init="auto")
        kmeans.fit(data_np)
        
        # 计算每个点到聚类中心的距离
        distances = np.linalg.norm(
            data_np.reshape(-1, 1, data_np.shape[1])
            - kmeans.cluster_centers_.reshape(1, -1, data_np.shape[1]),
            axis=2,
        )

        # 为每个聚类选择最近的点作为代表
        labels = kmeans.labels_
        core_indices = []
        for i in range(core_size):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) > 0:
                index = np.argmin(distances[cluster_indices, i])
                core_indices.append(cluster_indices[index])

        logger.info(f"KMeans core selection finished, selected {len(core_indices)} cores")
        return core_indices


