import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 将项目根目录添加到Python路径
root_dir = str(Path(__file__).parent.parent.parent)
sys.path.append(root_dir)

from dataset.config import CORE_SET_DATA_PATH, PCA_DATA_PATH, PROCESSED_DATA_DIR
from dataset.process.load_data import load_mnist


def process_and_save_data(n_components=10, eta=0.1, method="kmeans"):
    """
    处理MNIST数据并保存结果：
    1. 加载原始数据
    2. 进行PCA降维
    3. 选择核心集
    4. 保存所有结果
    """
    # 创建保存目录
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # 加载原始数据
    train_data, train_labels, test_data, test_labels = load_mnist()

    # PCA降维
    pca = PCA(n_components=n_components)
    train_data_pca = pca.fit_transform(train_data)
    test_data_pca = pca.transform(test_data)

    # 计算和保存解释方差比
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    print(f"PCA累计解释方差比: {cumulative_variance_ratio[-1]:.4f}")

    # 保存PCA结果
    pca_data = {
        "train_data": train_data_pca,
        "train_labels": train_labels,
        "test_data": test_data_pca,
        "test_labels": test_labels,
        "explained_variance_ratio": explained_variance_ratio,
    }

    with open(PCA_DATA_PATH, "wb") as f:
        pickle.dump(pca_data, f)

    print("PCA数据已保存")

    # 核心集选择
    n_samples = int(len(train_data_pca) * eta)

    if method == "kmeans":
        # 使用K-means聚类选择核心集
        kmeans = KMeans(
            n_clusters=n_samples,
            random_state=42,
            init='k-means++',  # 使用k-means++初始化方法
            n_init=10  # 运行10次取最好的结果
        )
        kmeans.fit(train_data_pca)

        # 选择离聚类中心最近的点作为核心集
        distances = kmeans.transform(train_data_pca)
        closest_points = np.argmin(distances, axis=0)

        core_data = train_data_pca[closest_points]
        core_labels = train_labels[closest_points]

    elif method == "random":
        # 随机选择核心集
        indices = np.random.choice(
            len(train_data_pca),
            n_samples,
            replace=False
        )
        core_data = train_data_pca[indices]
        core_labels = train_labels[indices]

    else:
        raise ValueError("method必须是'kmeans'或'random'之一")

    print(f"选择核心集大小: {len(core_data)}")

    # 保存核心集结果
    core_set_data = {
        "core_data": core_data,
        "core_labels": core_labels,
        "test_data": test_data_pca,
        "test_labels": test_labels,
        "method": method,
        "eta": eta,
    }

    with open(CORE_SET_DATA_PATH, "wb") as f:
        pickle.dump(core_set_data, f)

    print("核心集数据已保存")


if __name__ == "__main__":
    # 处理并保存数据
    process_and_save_data(n_components=10, eta=0.1, method="kmeans")