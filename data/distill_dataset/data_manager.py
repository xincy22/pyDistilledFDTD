import os

import hashlib
from diskcache import Cache
import numpy as np
import torch
from torch.utils.data import Dataset

from ..base_dataset.data_loader import load_core_set_data
from simulation import FDTDSimulator
from .config import DISTILL_DATA_DIR

class FDTDDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

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

    def get_train_dataset(self, regenerate: bool = False):
        train_key = f"{self.matrix_key}_train"

        if not regenerate and train_key in self.cache:
            data = self.cache[train_key]
            return FDTDDataset(data['inputs'], data['labels'])
        
        train_data, _, _, _ = load_core_set_data()
        inputs, labels = self._generate_dataset(train_data)

        self.cache[train_key] = {'inputs': inputs, 'labels': labels}

        return FDTDDataset(inputs, labels)

    def get_test_dataset(self):
        test_key = f"{self.matrix_key}_test"

        if test_key in self.cache:
            data = self.cache[test_key]
            return FDTDDataset(data['inputs'], data['labels'])

        _, _, test_data, _ = load_core_set_data()
        inputs, labels = self._generate_dataset(test_data)

        self.cache[test_key] = {'inputs': inputs, 'labels': labels}

        return FDTDDataset(inputs, labels)

    def _generate_dataset(self, data):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        simulator = FDTDSimulator(self.radius_matrix)
        
        x_batch = torch.tensor(data, device=device)
        y_batch = simulator(x_batch).detach().cpu()

        return x_batch, y_batch

    def save_series_model(self, model: torch.nn.Module):
        model_key = f"{self.matrix_key}_series_model"
        self.cache[model_key] = model.state_dict()

    def save_output_model(self, model: torch.nn.Module):
        model_key = f"{self.matrix_key}_output_model"
        self.cache[model_key] = model.state_dict()

    def load_series_model(self, model: torch.nn.Module):
        model_key = f"{self.matrix_key}_series_model"
        if model_key in self.cache: 
            model.load_state_dict(self.cache[model_key])

    def load_output_model(self, model: torch.nn.Module):
        model_key = f"{self.matrix_key}_output_model"
        if model_key in self.cache:
            model.load_state_dict(self.cache[model_key])

    def clear_cache(self):
        matrix_hash = self._calculate_matrix_hash()
        hash_index = self.cache['matrix_hash_index']

        if matrix_hash in hash_index:
            for key in hash_index[matrix_hash]:
                if torch.equal(self.radius_matrix, self.cache[key]):
                    self.cache.delete(key)
                    self.cache.delete(f"{key}_train")
                    self.cache.delete(f"{key}_test")
                    self.cache.delete(f"{key}_series_model")
                    self.cache.delete(f"{key}_output_model")
                    hash_index[matrix_hash].remove(key)

        self.cache['matrix_hash_index'] = hash_index

        if len(hash_index[matrix_hash]) == 0:
            del hash_index[matrix_hash]
            self.cache['matrix_hash_index'] = hash_index

        if len(self.cache) == 0:
            self.cache.clear()
