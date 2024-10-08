import os
import pickle

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

dir_path = os.path.dirname(os.path.abspath(__file__))
loader_cache_dir = os.path.join(dir_path, ".cache", "pca")
os.makedirs(loader_cache_dir, exist_ok=True)

os.environ["LOKY_MAX_CPU_COUNT"] = "4"


def load_mnist_data():
    """
    加载MNIST数据集并返回训练集和测试集的特征和标签。

    :return: 训练集和测试集的特征和标签。
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    mnist_train = datasets.MNIST(root=dir_path + "/.cache", train=True, download=True)
    mnist_test = datasets.MNIST(root=dir_path + "/.cache", train=False, download=True)

    x_train = mnist_train.data.numpy().astype(np.float32) / 255.0
    y_train = mnist_train.targets.numpy()
    x_test = mnist_test.data.numpy().astype(np.float32) / 255.0
    y_test = mnist_test.targets.numpy()

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    return x_train, y_train, x_test, y_test


class PCADataset(Dataset):
    """
    封装PCA降维后的数据集。

    :param data: 降维后的数据。
    :type data: torch.Tensor
    :param labels: 数据对应的标签。
    :type labels: torch.Tensor
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        """
        返回数据集的大小。

        :return: 数据集的样本数量。
        :rtype: int
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        返回指定索引的数据和标签。

        :param idx: 索引。
        :type idx: int
        :return: 数据和对应的标签。
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        return self.data[idx], self.labels[idx]


def pca_data_loader(n_components=10, batch_size=64):
    """
    对MNIST数据集进行PCA降维并返回降维后的DataLoader。

    :param n_components: PCA降维后的维度，默认为10。
    :type n_components: int
    :param batch_size: DataLoader的批次大小，默认为64。
    :type batch_size: int
    :return: PCA降维后的训练集和测试集的DataLoader。
    :rtype: tuple[DataLoader, DataLoader]
    """

    # 读取缓存文件加速
    cache_dir = os.path.join(loader_cache_dir, f"pca-components-{n_components}")
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = os.path.join(cache_dir, f"batch-size-{batch_size}.pkl")

    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as cache_file:
            print(f"Loading PCA data from {cache_filename}")
            return pickle.load(cache_file)

    x_train, y_train, x_test, y_test = load_mnist_data()

    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    # 对降维后的数据进行归一化，限制在0~1之间
    x_train_pca = (x_train_pca - x_train_pca.min(axis=0)) / (x_train_pca.max(axis=0) - x_train_pca.min(axis=0))
    x_test_pca = (x_test_pca - x_test_pca.min(axis=0)) / (x_test_pca.max(axis=0) - x_test_pca.min(axis=0))

    x_train_pca = torch.tensor(x_train_pca, dtype=torch.float64)
    x_test_pca = torch.tensor(x_test_pca, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    train_pca_dataset = PCADataset(x_train_pca, y_train)
    test_pca_dataset = PCADataset(x_test_pca, y_test)

    train_loader_pca = DataLoader(train_pca_dataset, batch_size=batch_size, shuffle=True)
    test_loader_pca = DataLoader(test_pca_dataset, batch_size=batch_size, shuffle=False)

    # 缓存文件
    with open(cache_filename, 'wb') as cache_file:
        print(f"Caching PCA data to {cache_filename}")
        pickle.dump((train_loader_pca, test_loader_pca), cache_file)

    return train_loader_pca, test_loader_pca


if __name__ == "__main__":
    train_loader, test_loader = pca_data_loader()
    for data, labels in train_loader:
        print(data.shape, labels.shape)
        break
