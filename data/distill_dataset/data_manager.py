import os
import hashlib
from diskcache import Cache
import numpy as np
import torch
from torch.utils.data import Dataset

from ..base_dataset.data_loader import load_core_set_data
from simulation import FDTDSimulator
from .config import DISTILL_DATA_DIR
from student.time_series import DataTransformer

class FDTDDataset(Dataset):
    def __init__(
            self, 
            data: torch.Tensor | np.ndarray, 
            labels: torch.Tensor | np.ndarray
    ):
        if isinstance(data, np.ndarray):
            self.data = torch.FloatTensor(data)
        else:
            self.data = data.float()

        if isinstance(labels, np.ndarray):
            self.labels = torch.FloatTensor(labels)
        else:
            self.labels = labels.float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class DistillDataManager:
    def __init__(self, radius_matrix: torch.Tensor | np.ndarray):
        if isinstance(radius_matrix, np.ndarray):
            self.radius_matrix = torch.FloatTensor(radius_matrix.copy())
        else:
            self.radius_matrix = radius_matrix.clone()

        if not os.path.exists(DISTILL_DATA_DIR):
            os.makedirs(DISTILL_DATA_DIR)
        self.cache = Cache(DISTILL_DATA_DIR)
        if 'matrix_hash_index' not in self.cache:
            self.cache['matrix_hash_index'] = {}

        self.matrix_key = self._get_matrix_key()

    def _calculate_matrix_hash(self):
        matrix_bytes = self.radius_matrix.numpy().tobytes()
        return hashlib.sha256(matrix_bytes).hexdigest()[:8]

    def _get_matrix_key(self):
        matrix_hash = self._calculate_matrix_hash()
        hash_index = self.cache['matrix_hash_index']

        if matrix_hash in hash_index:
            for matrix_key in hash_index[matrix_hash]:
                if torch.equal(self.radius_matrix, self.cache[matrix_key]):
                    return matrix_key

        matrix_key = f"matrix_{matrix_hash}"
        collision_count = 0
        while matrix_key in self.cache:
            collision_count += 1
            matrix_key = f"matrix_{matrix_hash}_{collision_count}"

        self.cache[matrix_key] = self.radius_matrix
        if matrix_hash not in hash_index:
            hash_index[matrix_hash] = []
        hash_index[matrix_hash].append(matrix_key)
        self.cache['matrix_hash_index'] = hash_index

        return matrix_key

    def _generate_dataset(self, data, mode='serial'):
        """生成数据集
        Args:
            data: 输入数据
            mode: 'serial' 或 'output'，决定数据处理方式
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        simulator = FDTDSimulator(self.radius_matrix)
        
        # 将数据批量转移到GPU
        x_batch = torch.tensor(data, device=device)
        y_batch = simulator(x_batch).detach()  # [batch_size, time_steps, ports]
        
        if mode == 'output':
            # 对时序数据取平均，得到每个端口的平均输出
            y_batch = y_batch.mean(dim=1)  # [batch_size, ports]
        
        # 如果是serial模式，需要使用data_transformer处理输入
        if mode == 'serial':
            data_transformer = DataTransformer(device=device)
            x_batch = data_transformer(x_batch)
        
        return x_batch.cpu(), y_batch.cpu()

    def get_train_dataset(self, mode='serial', regenerate: bool = False):
        """获取训练数据集
        Args:
            mode: 'serial' 或 'output'，决定数据处理方式
            regenerate: 是否重新生成数据
        """
        train_key = f"{self.matrix_key}_{mode}_train"

        if not regenerate and train_key in self.cache:
            data = self.cache[train_key]
            return FDTDDataset(data['inputs'], data['labels'])
        
        train_data, _, _, _ = load_core_set_data()
        inputs, labels = self._generate_dataset(train_data, mode=mode)

        self.cache[train_key] = {'inputs': inputs, 'labels': labels}

        return FDTDDataset(inputs, labels)

    def get_test_dataset(self, mode='serial'):
        """获取测试数据集
        Args:
            mode: 'serial' 或 'output'，决定数据处理方式
        """
        test_key = f"{self.matrix_key}_{mode}_test"

        if test_key in self.cache:
            data = self.cache[test_key]
            return FDTDDataset(data['inputs'], data['labels'])

        _, _, test_data, _ = load_core_set_data()
        inputs, labels = self._generate_dataset(test_data, mode=mode)

        self.cache[test_key] = {'inputs': inputs, 'labels': labels}

        return FDTDDataset(inputs, labels)

    def save_model(self, model: torch.nn.Module, mode='serial'):
        """保存模型
        Args:
            model: 要保存的模型
            mode: 'serial' 或 'output'，指定模型类型
        """
        model_key = f"{self.matrix_key}_{mode}_model"
        self.cache[model_key] = model.state_dict()

    def load_model(self, model: torch.nn.Module, mode='serial'):
        """加载模型
        Args:
            model: 要加载的模型
            mode: 'serial' 或 'output'，指定模型类型
        """
        model_key = f"{self.matrix_key}_{mode}_model"
        if model_key in self.cache:
            model.load_state_dict(self.cache[model_key])
        return model

    def clear_cache(self):
        """清理与当前矩阵相关的所有缓存数据"""
        matrix_hash = self._calculate_matrix_hash()
        hash_index = self.cache['matrix_hash_index']

        if matrix_hash in hash_index:
            for key in hash_index[matrix_hash]:
                if torch.equal(self.radius_matrix, self.cache[key]):
                    self.cache.delete(key)
                    self.cache.delete(f"{key}_serial_train")
                    self.cache.delete(f"{key}_serial_test")
                    self.cache.delete(f"{key}_output_train")
                    self.cache.delete(f"{key}_output_test")
                    self.cache.delete(f"{key}_serial_model")
                    self.cache.delete(f"{key}_output_model")
                    hash_index[matrix_hash].remove(key)

        self.cache['matrix_hash_index'] = hash_index

        if len(hash_index[matrix_hash]) == 0:
            del hash_index[matrix_hash]
            self.cache['matrix_hash_index'] = hash_index

        if len(self.cache) == 0:
            self.cache.clear()