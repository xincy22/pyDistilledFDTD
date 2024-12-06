# README

## 简介

本子包提供了用于加载和处理 MNIST 数据集的工具，包括 PCA 降维和核心集选择。

## 安装

确保已安装以下依赖：

- Python 3.x
- `numpy`
- `torch`
- `torchvision`
- `scikit-learn`

可以使用以下命令安装缺失的包：

```bash
pip install numpy torch torchvision scikit-learn
```

## 使用方法

### 导入模块

```python
from src.dataset import pca_data_loader, core_data_loader
```

### 加载 PCA 降维后的数据

```python
# 加载 PCA 降维后的训练集和测试集 DataLoader
train_loader_pca, test_loader_pca = pca_data_loader(n_components=10, batch_size=64)
```

- `n_components`: PCA 降维后的维度，默认为 10
- `batch_size`: DataLoader 的批次大小，默认为 64

### 加载核心集数据

```python
# 加载核心集的训练 DataLoader 和完整的测试 DataLoader
train_loader_core, test_loader = core_data_loader(
    eta=0.1,
    method='greedy',
    batch_size=64,
    n_components=10
)
```

- `eta`: 核心集与数据集的比例，0 < eta <= 1，默认为 0.1
- `method`: 核心集选择方法，'greedy' 或 'kmeans'，默认为 'greedy'
- `batch_size`: DataLoader 的批次大小，默认为 64
- `n_components`: PCA 降维后的维度，默认为 10

### 示例

以下是一个完整的示例，展示如何使用核心集的 DataLoader 进行训练：

```python
import torch
from src.dataset import core_data_loader

# 加载核心集 DataLoader
train_loader_core, test_loader = core_data_loader(
  eta=0.1,
  method='greedy',
  batch_size=64,
  n_components=10
)

# 定义简单的模型
model = torch.nn.Linear(10, 10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 训练模型
for epoch in range(5):
  for data, labels in train_loader_core:
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
  print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# 测试模型
correct = 0
total = 0
with torch.no_grad():
  for data, labels in test_loader:
    outputs = model(data)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print(f'Accuracy on test set: {100 * correct / total}%')
```

## 函数说明

### `pca_data_loader(n_components=10, batch_size=64)`

- 功能：对 MNIST 数据集进行 PCA 降维并返回降维后的训练集和测试集 DataLoader。
- 参数：
  - `n_components`：PCA 降维后的维度。
  - `batch_size`：DataLoader 的批次大小。
- 返回：
  - `train_loader_pca`：训练集 DataLoader。
  - `test_loader_pca`：测试集 DataLoader。

### `core_data_loader(eta=0.1, method='greedy', batch_size=64, n_components=10)`

- 功能：加载核心集的训练 DataLoader 和完整的测试 DataLoader。
- 参数：
  - `eta`：核心集与数据集的比例。
  - `method`：核心集选择方法，'greedy' 或 'kmeans'。
  - `batch_size`：DataLoader 的批次大小。
  - `n_components`：PCA 降维后的维度。
- 返回：
  - `train_loader_core`：核心集训练 DataLoader。
  - `test_loader`：测试集 DataLoader。

