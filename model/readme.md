# README

## 简介

本子包实现了一个从有限差分时域（FDTD）仿真到神经网络的**知识蒸馏**框架，旨在通过神经网络模型高效地近似和替代复杂的 FDTD 仿真，适用于实时需求。主要功能包括：

1. **FDTD 仿真**：利用内置的 `fdtd` 库进行电磁波仿真，支持配置仿真区域、介电常数、源和探测器等。
2. **知识蒸馏**：将 FDTD 仿真的输出作为训练数据，训练基于 LSTM 的神经网络模型，学习并模拟仿真结果。
3. **高效推理**：一旦神经网络模型经过训练，可以用它替代传统的 FDTD 仿真，实现快速、高效的推理，适用于实时需求。

## 安装

本包依赖以下软件和库：

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- TQDM
- PyTorch

安装所需的依赖库：

```bash
pip install numpy scipy matplotlib tqdm torch
```

## 使用方法

本子包的主要使用流程包括：

1. **初始化 `DistillModel`**：创建一个 `DistillModel` 实例，传入仿真区域的 `radius_matrix` 和学生模型类（如 `StudentOutputModel` 或 `StudentSequenceModel`）。
2. **配置仿真源和介电常数**：在 `DistillModel` 中设置电磁源和介电常数分布。
3. **运行仿真或推理**：调用 `sim()` 方法运行 FDTD 仿真，或直接调用模型进行前向推理。
4. **训练学生模型**：使用 FDTD 仿真数据训练学生模型，使其能够学习并模拟仿真结果。

### 目录结构

```
model/
├── fdtd/                   # 内置的 fdtd 库
├── distill_model.py        # 核心的 DistillModel 类
├── student_model.py        # 学生模型，包括 StudentOutputModel 和 StudentSequenceModel
└── __init__.py             # 包的初始化文件
```

### 示例代码

以下示例包括两个部分：运行仿真和训练学生模型。

#### 1. 运行仿真

```python
import torch
import numpy as np
from model import DistillModel, StudentOutputModel

# 定义仿真区域的随机半径矩阵
radius_matrix = np.random.rand(10, 10)

# 初始化蒸馏模型，传入学生模型类
distill_model = DistillModel(radius_matrix, StudentOutputModel)

# 设置模型为只使用 FDTD 仿真
distill_model.set_simulation_mode(fdtd=True, lstm=False)
distill_model.eval()

# 生成一个输入源（每个端口的相位值），假设有 10 个端口
batch_size = 1
ports = 10
source_input = torch.rand(batch_size, ports)  # 随机生成相位输入

# 运行 FDTD 仿真并获得输出
output, output_by_time = distill_model.sim(source_input)

print("FDTD 仿真输出（端口总强度）：", output)
print("FDTD 仿真输出（按时间步长）：", output_by_time)
```

#### 2. 训练学生模型

```python
import torch
import time
import os
from tqdm import tqdm
import numpy as np
from dataset import core_data_loader
from model import DistillModel, StudentSequenceModel

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
train_loader, test_loader = core_data_loader(eta=0.01, batch_size=4)

print(f"训练数据大小: {len(train_loader.dataset)}")
print(f"测试数据大小: {len(test_loader.dataset)}")

# 定义仿真区域的随机半径矩阵
radius_matrix = np.random.rand(10, 10)

# 初始化蒸馏模型，传入学生模型类
model = DistillModel(radius_matrix, StudentSequenceModel).to(device)

# 定义优化器
optimizer = torch.optim.Adam(model.student_model.parameters(), lr=0.01)

# 训练学生模型
epochs = 10
model.train()
model.set_simulation_mode(fdtd=True, lstm=False)  # 训练时需要运行 FDTD 仿真

for epoch in range(epochs):
    running_loss = 0.0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            
            # 计算损失
            loss = model(inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.update(1)
            pbar.set_postfix(loss=running_loss / (pbar.n or 1))
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# 保存模型
save_path = f"student_model_{time.strftime('%Y-%m-%d-%H-%M-%S')}.pth"
torch.save({
    'radius_matrix': radius_matrix,
    'student_model_state_dict': model.student_model.state_dict(),
}, save_path)
print(f"模型已保存到: {save_path}")

# 测试模型
model.eval()
model.set_simulation_mode(fdtd=True, lstm=True)
criterion = torch.nn.MSELoss()
total_loss = 0.0

with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        fdtd_output, lstm_output = model(inputs)
        loss = criterion(lstm_output, fdtd_output)
        total_loss += loss.item()

print(f"测试集平均损失: {total_loss / len(test_loader)}")
```

### 修改和扩展

在克隆仓库后，您可以根据自己的需求修改以下内容，以适应不同的应用场景：

- **更改仿真区域**：修改 `radius_matrix`，以生成不同的介电常数分布，创建不同的介质结构。
- **调整仿真参数**：在 `config.py` 中修改仿真的全局参数，如网格大小、时间步长、波长等。
- **自定义学生模型**：在 `student_model.py` 中定义新的神经网络模型，替换或扩展现有的学生模型。
- **修改数据扩展方法**：更改 `data_expand` 函数的扩展方法，以尝试不同的输入数据格式。

## 模块解释

本子包主要由以下模块组成：

### 1. `DistillModel`

#### 功能概述

`DistillModel` 是核心类，用于将 FDTD 仿真与神经网络学生模型相结合，实现知识蒸馏。

- **FDTD 仿真初始化**：使用内置的 `fdtd` 库构建仿真网格，设置边界条件、波导、源和探测器。
- **设置介电常数**：根据提供的 `radius_matrix` 配置仿真区域的介电常数分布，生成特定的介质结构。
- **设置源**：根据输入源值（相位信息），设置仿真中的电磁源。
- **运行仿真**：调用 `sim()` 方法运行 FDTD 仿真，并获取仿真结果。
- **知识蒸馏**：通过比较 FDTD 仿真输出和学生模型预测，计算损失并训练学生模型。
- **仿真缓存**：引入缓存机制，避免重复计算相同的仿真，提升效率。

#### 核心方法

- **`__init__(self, radius_matrix, student_model_class)`**：初始化模型，设置介电常数矩阵和学生模型。
  
  ```python
  distill_model = DistillModel(radius_matrix, StudentOutputModel)
  ```
  
- **`grid_init(self)`**：初始化仿真网格，添加边界条件、波导和探测器。
- **`set_source(self, source)`**：设置仿真中的电磁源，相位信息由输入 `source` 决定。
  
  ```python
  distill_model.set_source(source_input)
  ```
  
- **`set_permittivity(self)`**：根据 `radius_matrix` 设置仿真区域的介电常数分布。
- **`sim(self, source)`**：对输入源运行仿真，支持缓存机制，返回仿真结果。

  ```python
  output, output_by_time = distill_model.sim(source_input)
  ```
  
- **`forward(self, source)`**：前向传播方法，根据训练模式和仿真模式返回相应的结果或损失。
- **`set_simulation_mode(self, fdtd=True, lstm=False)`**：设置模型的仿真模式，选择使用 FDTD 仿真、学生模型或两者。

  ```python
  distill_model.set_simulation_mode(fdtd=False, lstm=True)
  ```

#### 原理分析

`DistillModel` 将复杂的 FDTD 仿真过程封装起来，使用户可以方便地运行仿真和训练神经网络模型。

- **FDTD 仿真**：利用 `fdtd` 库进行电磁场仿真，获取高精度的仿真结果，作为知识蒸馏的教师模型。
- **知识蒸馏**：通过让学生模型（神经网络）学习 FDTD 仿真的输出，使其能够快速近似仿真结果，实现高效推理。
- **缓存机制**：通过对仿真结果进行缓存，避免重复计算，提高仿真效率。

### 2. 学生模型

#### `StudentOutputModel`

- **功能**：学习并预测各端口的总输出强度。
- **结构**：包含 LSTM 层和全连接层，处理时间序列数据并输出预测结果。
- **损失函数**：使用均方误差（MSE）计算模型预测与 FDTD 仿真输出之间的差异。

##### 使用示例

```python
from model import StudentOutputModel

# 初始化学生模型
student_model = StudentOutputModel(input_size=ports, hidden_size=128, output_size=ports)

# 前向传播
predictions = student_model(source_input)
```

#### `StudentSequenceModel`

- **功能**：学习并预测各时间步的端口输出强度序列。
- **结构**：与 `StudentOutputModel` 类似，但输出为时间序列数据。
- **损失函数**：使用均方误差（MSE）计算模型预测的时间序列与仿真输出的时间序列之间的差异。

##### 使用示例

```python
from model import StudentSequenceModel

# 初始化学生模型
student_model = StudentSequenceModel(input_size=ports, hidden_size=128, output_size=ports)

# 前向传播
predictions = student_model(source_input)
```

#### 原理分析

- **LSTM 网络**：由于 FDTD 仿真输出的是时间序列数据，LSTM 网络能够有效地捕获时序特征，适合处理这种数据。
- **数据扩展**：在模型的前向传播中，输入数据会根据一定的方法（如重复、正弦函数）扩展到指定的时间步长，以适应 LSTM 的输入要求。

### 3. 内置 `fdtd` 库

本子包内置了 `fdtd` 库，无需用户额外安装。`fdtd` 库提供了强大的 FDTD 仿真能力，支持多种配置和组件。

#### 主要功能

- **多种后端支持**：支持 `numpy` 和 `torch` 后端，方便在 CPU 和 GPU 上运行。
- **灵活的仿真区域配置**：可以设置仿真区域的形状、大小、介电常数等。
- **丰富的组件**：提供了源、探测器、边界条件、物体等组件，方便构建复杂的仿真场景。
- **高效的仿真性能**：优化的算法和实现，保证了仿真的效率。

#### 仿真区域初始化

- **仿真后端**：可以指定后端类型，如 `torch.cuda.float64` 等。

  ```python
  fdtd.set_backend("torch")
  ```

- **仿真区域**：通过 `Grid` 类定义，包括形状、网格间距、介电常数等。

  ```python
  grid = fdtd.Grid(shape=(Nx, Ny, Nz), grid_spacing=dx)
  ```

- **边界条件**：支持 PML（完美匹配层）、周期性边界等，防止波的反射。

  ```python
  # 添加 PML 边界条件
  grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
  grid[-10:, :, :] = fdtd.PML(name="pml_xhigh")
  ```

- **波源和探测器**：提供多种类型的源和探测器，支持点源、线源、平面波源等。

  ```python
  # 添加线源
  grid[20, 33:37, 0] = fdtd.LineSource(name="source")

  # 添加线探测器
  grid[30, 33:37, 0] = fdtd.LineDetector(name="detector")
  ```

## 更多信息

- **源码**：请参阅源码文件 `distill_model.py` 和 `student_model.py`，其中包含详细的代码注释和文档字符串。
- **官方文档**：更多关于 `fdtd` 库的使用，请参考官方文档：https://fdtd.readthedocs.org/
