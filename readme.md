# README

## 简介

本项目的核心目标是通过知识蒸馏技术对电磁波的时域有限差分（FDTD）模拟进行加速。传统的FDTD仿真虽然能够高精度地模拟电磁波的传播，但其计算成本极高，尤其在复杂几何结构和多维设计问题上表现尤为明显。

为了解决这一问题，本项目提出了基于知识蒸馏的仿真加速技术。通过构建神经网络LSTM模型，学习复杂的FDTD仿真结果，使其在保持一定精度的前提下，实现对仿真过程的高效替代，从而大幅度减少计算成本。

## 项目结构

```
├── dataset/                # 数据处理和加载模块
│   ├── __init__.py
│   ├── core_dataset.py     # 核心集选择和数据加载
│   └── pca_dataset.py      # PCA 降维和数据加载
├── model/                  # 知识蒸馏和神经网络模型
│   ├── __init__.py
│   ├── distill_model.py    # 知识蒸馏核心模型
│   ├── student_model.py    # 学生模型（神经网络）
│   └── fdtd/               # 内置的 FDTD 库
├── config.py               # 全局配置文件
└── train.py                # 训练脚本示例
```

## 知识蒸馏的原理

FDTD本质上是一个具有时序数据处理功能的神经网络。仿真区域中的源就是该时序神经网络的输入，仿真区域中的传感器就是该时序神经网络的输出。仿真区域中的介质就是该时序神经网络的参数。仿真区域中的探测器就是该时序神经网络的输出。

该时序网络和LSTM具有一定的相似性，对于2D FDTD，其隐藏层可以使用三组量 $H_x,H_y,E_z$ 来表示，单个隐藏层的更新公式为

$$
\left\{
\begin{aligned}
    &\begin{aligned}
        &H_x^{n+1/2}\left(i,j+\frac 12\right) = CP(m)\cdot H_x^{n-1/2}\left(i,j+\frac12\right)-CQ(m)\cdot \frac{E_z^n(i,j+1)-E_z^n(i,j)}{\Delta y}\\
        &H_y^{n+1/2}\left(i+\frac 12,j\right) = CP(m)\cdot H_y^{n-1/2}\left(i+\frac12,j\right)-CQ(m)\cdot \frac{E_z^n(i+1,j)-E_z^n(i,j)}{\Delta x}
    \end{aligned}\\
    \\
    &\begin{aligned}
        E_z^{n+1}(i,j)=&\;CA(m)\cdot E_z^n(i,j)\\
        +&CB(m)\cdot \left[
            \begin{aligned}
                &\frac{H_y^{n+1/2}\left(i+\frac12,j\right)-H_y^{n+1/2}\left(i-\frac12,j\right)}{\Delta x}\\
                &-\frac{H_x^{n+1/2}\left(i,j+\frac12\right)-H_x^{n+1/2}\left(i,j-\frac12\right)}{\Delta y}-J_{source_z}^{n+1/2}
            \end{aligned}
        \right]
    \end{aligned}
\end{aligned}
\right.
$$

其中 $CP(m),CQ(m),CA(m),CB(m)$ 是介质参数，$J_{source_z}^{n+1/2}$ 是源项。

因此本项目考虑使用LSTM对这个大网络进行知识蒸馏，由于架构的相似性，我们可以认为LSTM可以学习到FDTD的特征，从而实现对FDTD的加速。

## 物理模型介绍

本项目仿真的物理模型是一个 $6.25\mathrm{\mu m} \times 6.25\mathrm{\mu m}$ 的方形区域，仿真的网格的大小为 $25\mathrm{n m}$ ，时间步长约为 $5.84\times 10^{-17}\mathrm s$，总的时间步为 $1000$ 步。

仿真区域在 $(75:325, 75:325)$ ，有 $10$ 个端口的输入和 $10$ 个端口的输出。在仿真区域中有 $10\times 10$ 的圆孔，圆孔半径的范围在 $[75\mathrm{nm}, 250\mathrm{nm}]$，整个仿真区域的示意图如下：

![](https://raw.githubusercontent.com/xincy22/MyImages/main/img/20241010153007.png)

## 使用方法

本项目使用上述的物理模型部署手写数字识别任务，将Detector收集到的光强作为各个标签的置信度。

### 准备数据

仿真区域仅有 $10$ 个端口的输入，在尝试运行在MNIST数据集之前，需要对数据集进行预处理，将其转换为 $10$ 个端口的输入。

在 `dataset` 包中提供了数据处理和加载模块，包括 PCA 降维和核心集选择。主要提供了以下两个函数：

- `pca_data_loader(n_components=10, batch_size=64)`: 加载 PCA 降维后的训练集和测试集 DataLoader
- `core_data_loader(eta=0.1, method='greedy', batch_size=64, n_components=10)`: 加载核心集的训练 DataLoader 和完整的测试 DataLoader

使用示例：

```python
from dataset import pca_data_loader, core_data_loader

# 加载 PCA 降维后的训练集和测试集 DataLoader
train_loader_pca, test_loader_pca = pca_data_loader(n_components=10, batch_size=64)

# 加载核心集的训练 DataLoader 和完整的测试 DataLoader
train_loader_core, test_loader = core_data_loader(
    eta=0.1,
    method='greedy',
    batch_size=64,
    n_components=10
)
```

详细内容请参考 `dataset` 包中的 `readme.md`。

### 训练模型

在 `model` 包中提供了知识蒸馏和神经网络模型，主要提供了以下三个类：

- `DistillModel`: 知识蒸馏核心模型
- `StudentOutputModel`: 直接学习最终输出的学生模型
- `StudentSequenceModel`：学习时序输出的学生模型

使用示例：

```python
from model import DistillModel, StudentSequenceModel

# 定义知识蒸馏核心模型和优化器
model = DistillModel(radius_matrix, StudentSequenceModel).to(device)
optimizer = torch.optim.Adam(model.student_model.parameters(), lr=0.01)

# 训练Student模型
epochs = 10
with tqdm(total=epochs * len(train_loader)) as pbar:
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        with torch.enable_grad():
            for inputs, _ in train_loader:
                inputs = inputs.to(device)
                
                loss = model(inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=running_loss)
```

详细内容请参考 `model` 包中的 `readme.md`。

### 前向仿真

可以直接使用 `fdtd` 库来进行前向仿真，也可以使用知识蒸馏后的模型来进行快速仿真。

使用示例：
```python
# 退出训练模式
model.eval()

# 使用fdtd库进行仿真
model.set_simulation_mode(fdtd=True, lstm=False)
print(model(inputs))

# 使用知识蒸馏后的模型进行仿真
model.set_simulation_mode(fdtd=False, lstm=True)
print(model(inputs))

# 同时使用两种模式进行仿真
model.set_simulation_mode(fdtd=True, lstm=True)
print(model(inputs))
```

详细内容请参考 `model` 包中的 `readme.md`。