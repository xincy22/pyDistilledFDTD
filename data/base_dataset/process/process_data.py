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

from base_dataset.config import CORE_SET_DATA_PATH, PCA_DATA_PATH, PROCESSED_DATA_DIR
from base_dataset.process.load_data import load_mnist


def process_and_save_data(n_components=10, eta=0.1, method="kmeans"):
    """
    处理MNIST数据并保存结果：
    1. 加载原始数据
    2. 进行PCA降维
    3. 对训练集和测试集分别选择核心集
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

    def select_core_set(data, labels, n_samples, method):
        """辅助函数：选择核心集"""
        if method == "kmeans":
            kmeans = KMeans(
                n_clusters=n_samples,
                random_state=42,
                init='k-means++',
                n_init=10
            )
            kmeans.fit(data)
            
            distances = kmeans.transform(data)
            closest_points = np.argmin(distances, axis=0)
            
            return data[closest_points], labels[closest_points]
            
        elif method == "random":
            indices = np.random.choice(len(data), n_samples, replace=False)
            return data[indices], labels[indices]
        
        else:
            raise ValueError("method必须是'kmeans'或'random'之一")

    # 计算训练集和测试集的核心集大小
    n_train_samples = int(len(train_data_pca) * eta)
    n_test_samples = int(len(test_data_pca) * eta)

    # 分别选择训练集和测试集的核心集
    core_train_data, core_train_labels = select_core_set(
        train_data_pca, train_labels, n_train_samples, method
    )
    core_test_data, core_test_labels = select_core_set(
        test_data_pca, test_labels, n_test_samples, method
    )

    print(f"训练集核心集大小: {len(core_train_data)}")
    print(f"测试集核心集大小: {len(core_test_data)}")

    # 保存核心集结果
    core_set_data = {
        "core_train_data": core_train_data,
        "core_train_labels": core_train_labels,
        "core_test_data": core_test_data,
        "core_test_labels": core_test_labels,
        "method": method,
        "eta": eta,
    }

    with open(CORE_SET_DATA_PATH, "wb") as f:
        pickle.dump(core_set_data, f)

    print("核心集数据已保存")


if __name__ == "__main__":
    # 处理并保存数据
    process_and_save_data(n_components=10, eta=0.02, method="kmeans")
