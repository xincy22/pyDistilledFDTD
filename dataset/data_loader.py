import os
import pickle
from .config import PCA_DATA_PATH, CORE_SET_DATA_PATH


def load_pca_data():
    """
    加载PCA降维后的MNIST数据。

    Returns:
        tuple: (train_data, train_labels, test_data, test_labels)
    """
    if not os.path.exists(PCA_DATA_PATH):
        raise FileNotFoundError(
            "PCA数据文件不存在，请先运行process/process_data.py处理数据"
        )
    
    with open(PCA_DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    return (
        data['train_data'],
        data['train_labels'],
        data['test_data'],
        data['test_labels']
    )


def load_core_set_data():
    """
    加载核心集数据。

    Returns:
        tuple: (core_data, core_labels, test_data, test_labels)
    """
    if not os.path.exists(CORE_SET_DATA_PATH):
        raise FileNotFoundError(
            "核心集数据文件不存在，请先运行process/process_data.py处理数据"
        )
    
    with open(CORE_SET_DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    return (
        data['core_train_data'],
        data['core_train_labels'],
        data['core_test_data'],
        data['core_test_labels']
    )
