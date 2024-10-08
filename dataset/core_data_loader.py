import os
import pickle

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from torch.utils.data import DataLoader

from .pca_data_loader import pca_data_loader, PCADataset

dir_path = os.path.dirname(os.path.abspath(__file__))
loader_cache_dir = os.path.join(dir_path, ".cache", "core")

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

def select_core_set_greedy(X, eta=0.1):
    """
    使用贪心算法选择核心集。

    逐步选择与当前核心集最远的数据点，确保核心集中的样本之间具有最大的差异性。

    :param X: 数据特征，numpy数组形状为(N, D)。
    :type X: np.ndarray
    :param eta: 核心集与数据集的比例，0 < eta <= 1，默认为0.1。
    :type eta: float
    :return: 核心集的索引列表。
    :rtype: list[int]
    """
    N = X.shape[0]
    core_size = max(1, int(eta * N))
    core_indices = []

    # 随机选择第一个点
    first_idx = np.random.randint(0, N)
    core_indices.append(first_idx)

    # 初始化最小距离数组
    min_dist = np.linalg.norm(X - X[first_idx], axis=1)

    for _ in range(1, core_size):
        # 选择距离当前核心集最远的点
        next_idx = np.argmax(min_dist)
        core_indices.append(next_idx)

        # 更新最小距离
        dist = np.linalg.norm(X - X[next_idx], axis=1)
        min_dist = np.minimum(min_dist, dist)

    return core_indices


def select_core_set_kmeans(X, eta=0.1):
    """
    使用KMeans算法选择核心集。

    对数据进行KMeans聚类，并选择每个簇中最接近簇中心的点作为核心集。

    :param X: 数据特征，numpy数组形状为(N, D)。
    :type X: np.ndarray
    :param eta: 核心集与数据集的比例，0 < eta <= 1，默认为0.1。
    :type eta: float
    :return: 核心集的索引列表。
    :rtype: list[int]
    """
    N = X.shape[0]
    core_size = max(1, int(eta * N))

    # 使用K-Means聚类
    kmeans = KMeans(n_clusters=core_size, random_state=42, n_init='auto')
    kmeans.fit(X)

    # 选择每个簇中最接近簇中心的点
    core_indices = []
    for i in range(core_size):
        cluster_center = kmeans.cluster_centers_[i].reshape(1, -1)
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_points = X[cluster_indices]
        closest, _ = pairwise_distances_argmin_min(cluster_center, cluster_points)
        core_idx = cluster_indices[closest[0]]
        core_indices.append(core_idx)

    return core_indices


def select_core_set(X, eta=0.1, method='greedy'):
    """
    核心集选择的统一接口。

    根据指定的方法选择核心集。

    :param X: 数据特征，numpy数组形状为(N, D)。
    :type X: np.ndarray
    :param eta: 核心集与数据集的比例，0 < eta <= 1，默认为0.1。
    :type eta: float
    :param method: 选择方法，'greedy' 或 'kmeans'，默认为'greedy'。
    :type method: str
    :return: 核心集的索引列表。
    :rtype: list[int]
    """
    if method == 'greedy':
        return select_core_set_greedy(X, eta)
    elif method == 'kmeans':
        return select_core_set_kmeans(X, eta)
    else:
        raise ValueError("Unsupported method. Choose 'greedy' or 'kmeans'.")


def core_data_loader(eta=0.1, method='greedy', batch_size=64, n_components=10):
    """
    完整流程：加载数据 -> PCA降维 -> 核心集选择 -> 返回核心集的DataLoader。

    :param eta: 核心集与数据集的比例，0 < eta <= 1，默认为0.1。
    :type eta: float
    :param method: 核心集选择方法，'greedy' 或 'kmeans'，默认为'greedy'。
    :type method: str
    :param batch_size: DataLoader的批次大小，默认为64。
    :type batch_size: int
    :param n_components: PCA降维后的维度，默认为10。
    :type n_components: int
    :return: 核心集的训练DataLoader和完整的测试DataLoader。
    :rtype: tuple[DataLoader, DataLoader]
    """
    if not (0 < eta <= 1):
        raise ValueError("Invalid eta. eta should be in (0, 1].")
    # 读取缓存文件加速
    cache_dir = os.path.join(loader_cache_dir, f"{method}", f"pca-components-{n_components}")
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = os.path.join(cache_dir, f"batch-size-{batch_size}-eta-{eta}.pkl")

    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as cache_file:
            print(f"Loading core data from {cache_filename}")
            return pickle.load(cache_file)

    # 加载PCA降维后的数据
    train_loader_pca, test_loader_pca = pca_data_loader(n_components=n_components, batch_size=batch_size)
    x_train = train_loader_pca.dataset.data.numpy()
    y_train = train_loader_pca.dataset.labels.numpy()

    x_test = test_loader_pca.dataset.data.numpy()
    y_test = test_loader_pca.dataset.labels.numpy()

    # 核心集选择
    core_indices = select_core_set(x_train, eta=eta, method=method)
    print(f"使用 {method} 方法选择了 {len(core_indices)} 个核心样本。")

    # 创建核心集的子集
    x_train_core = x_train[core_indices]
    y_train_core = y_train[core_indices]

    train_core_dataset = PCADataset(
        torch.tensor(x_train_core, dtype=torch.float64),
        torch.tensor(y_train_core, dtype=torch.int64)
    )
    test_dataset = PCADataset(
        torch.tensor(x_test, dtype=torch.float64),
        torch.tensor(y_test, dtype=torch.int64)
    )

    train_loader_core = DataLoader(train_core_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 缓存文件
    with open(cache_filename, 'wb') as cache_file:
        print(f"Saving core data to {cache_filename}")
        pickle.dump((train_loader_core, test_loader), cache_file)

    return train_loader_core, test_loader


if __name__ == "__main__":
    train_core_loader, test_loader = core_data_loader(method='kmeans')
    for data, labels in train_core_loader:
        print(data.shape, labels.shape)
        break
