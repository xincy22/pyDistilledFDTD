import torch
import numpy as np
import matplotlib.pyplot as plt

# 模拟 cfg 配置对象
class Config:
    simulation_step = 10
    WAVELENGTH = 1550e-9  # 假设波长为 3 cm
    SPEED_LIGHT: float = 299_792_458.0

# 创建模拟 grid 类，它包含时间步长属性
class Grid:
    time_step = 1e-9  # 假设时间步长为 1 纳秒

# 测试 data_expand 方法的类
class DataExpander:
    def __init__(self):
        self.grid = Grid()

    def data_expand(self, source, method='repeat'):
        if method == "repeat":
            return source.unsqueeze(1).expand(-1, cfg.simulation_step, -1)
        elif method == "gaussian":
            mean = (cfg.simulation_step - 1) / 2.0
            std_dev = (cfg.simulation_step - 1) / 6.0

            t = torch.arange(cfg.simulation_step, device=source.device, dtype=source.dtype)
            gaussian_curve = torch.exp(-((t - mean) ** 2) / (2 * std_dev ** 2))
            gaussian_curve = gaussian_curve / gaussian_curve.sum()
            expanded_value = gaussian_curve.view(1, -1, 1) * source.unsqueeze(1)
            return expanded_value * 50
        elif method == "sin":
            period = cfg.WAVELENGTH / cfg.SPEED_LIGHT
            omega = 2 * torch.pi / period

            t = torch.arange(cfg.simulation_step).view(1, cfg.simulation_step, 1) * self.grid.time_step
            phase_shift = source.unsqueeze(1) * torch.pi
            sin_wave = torch.sin(omega * t + phase_shift)
            return sin_wave


# 创建 cfg 实例
cfg = Config()

# 创建一个 DataExpander 实例
expander = DataExpander()

# 输入数据：batch_size=1，ports=3 的张量，元素值在[0, 1]之间
source = torch.tensor([[0.2, 0.5, 0.8]])

# 扩展后的数据 (sin 方法)
expanded_sin = expander.data_expand(source, method='sin')

# 仿真步长用于绘图的时间轴
time_steps = np.linspace(0, cfg.simulation_step * expander.grid.time_step, cfg.simulation_step)

# 绘制正弦波和扩展后的数据
ports_labels = ['Port 1', 'Port 2', 'Port 3']
for port_idx in range(source.size(1)):
    plt.figure(figsize=(10, 6))

    # 绘制标准正弦函数（连续的正弦曲线）
    omega = 2 * np.pi / (cfg.WAVELENGTH / cfg.SPEED_LIGHT)
    t_values = np.linspace(0, cfg.simulation_step * expander.grid.time_step, 10000)  # 连续时间值用于绘制完整正弦波
    sin_function = np.sin(omega * t_values + source[0, port_idx].item() * np.pi)
    plt.plot(t_values, sin_function, label=f'Sin function for {ports_labels[port_idx]}', color='blue')

    # 绘制扩展后的散点
    plt.scatter(time_steps, expanded_sin[0, :, port_idx].cpu().numpy(), color='red', label=f'Expanded Data for {ports_labels[port_idx]}')

    plt.title(f'Sin Expansion and Selected Points - {ports_labels[port_idx]}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.show()