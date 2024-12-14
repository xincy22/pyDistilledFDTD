import torch
from typing import Optional

import torch
import numpy as np
from typing import Optional
from simulation.config import WAVELENGTH, SPEED_LIGHT, simulation_step

class SignalExpansion:
    """将相位信号扩展为时序信号的类"""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        初始化信号扩展器

        Args:
            device: 计算设备
        """
        self.wavelength = WAVELENGTH
        self.speed_light = SPEED_LIGHT
        self.time_steps = simulation_step
        self.device = device if device is not None else torch.device('cpu')
        
        # 计算周期和频率
        self.period = self.wavelength / self.speed_light
        self.frequency = 1.0 / self.period
        
        # 生成时间步
        self.time = torch.arange(
            self.time_steps,
            device=self.device,
            dtype=torch.float32
        )
        
    def expand(self, phase: torch.Tensor) -> torch.Tensor:
        """
        将相位信号扩展为时序信号

        Args:
            phase: 相位信号，形状为 (batch_size, ports) 或 (ports,)

        Returns:
            torch.Tensor: 扩展后的时序信号，形状为 (batch_size, time_steps, ports) 或 (time_steps, ports)
        """
        # 确保输入是2D张量
        if phase.dim() == 1:
            phase = phase.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # 将相位转换到设备上
        phase = phase.to(self.device)
        
        # 计算角频率
        omega = 2 * np.pi / self.period
        
        # 扩展维度以进行广播
        t = self.time.view(1, -1, 1)  # (time_steps, 1)
        phase_shift = phase.unsqueeze(1) * np.pi  # (batch_size, 1, ports)

        # 生成正弦信号
        signal = torch.sin(
            omega * t + phase_shift  
        ) # (batch_size, time_steps, ports)
        
        # 如果输入是1D的，返回2D结果
        if squeeze_output:
            signal = signal.squeeze(0)  # (time_steps, ports)
            
        return signal

    def __call__(self, phase: torch.Tensor) -> torch.Tensor:
        """方便直接调用类实例来扩展信号"""
        return self.expand(phase)

class DataTransformer:
    """处理输入数据转换的类"""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        初始化数据转换器

        Args:
            device: 计算设备
        """
        self.device = device if device is not None else torch.device('cpu')
        self.signal_expander = SignalExpansion(device=self.device)
        
    def _add_zero_channel(self, signal: torch.Tensor) -> torch.Tensor:
        """
        在信号中添加零值通道

        Args:
            signal: 输入信号，形状为 (batch_size, time_steps, ports) 或 (time_steps, ports)

        Returns:
            torch.Tensor: 添加零值通道后的信号，形状为 (batch_size, time_steps, ports+1) 或 (time_steps, ports+1)
        """
        if signal.dim() == 2:
            # 如果是2D输入 (time_steps, ports)
            zero_channel = torch.zeros(signal.size(0), 1, device=signal.device)
            return torch.cat([signal, zero_channel], dim=1)
        else:
            # 如果是3D输入 (batch_size, time_steps, ports)
            zero_channel = torch.zeros(signal.size(0), signal.size(1), 1, device=signal.device)
            return torch.cat([signal, zero_channel], dim=2)
    
    def transform(self, phase: torch.Tensor) -> torch.Tensor:
        """
        转换输入相位数据为LSTM可用的输入格式

        Args:
            phase: 输入相位，形状为 (batch_size, ports) 或 (ports,)

        Returns:
            torch.Tensor: 转换后的信号，形状为 (batch_size, time_steps, ports+1) 或 (time_steps, ports+1)
        """
        # 记录输入维度
        is_batch = phase.dim() == 2
        if not is_batch:
            phase = phase.unsqueeze(0)
            
        # 将相位扩展为时序信号
        signal = self.signal_expander(phase)  # (batch_size, time_steps, ports)
        
        # 添加零值通道
        signal = self._add_zero_channel(signal)  # (batch_size, time_steps, ports+1)
        
        # 如果输入不是批量的，去掉批量维度
        if not is_batch:
            signal = signal.squeeze(0)
            
        return signal
    
    def __call__(self, phase: torch.Tensor) -> torch.Tensor:
        """方便直接调用类实例来转换数据"""
        return self.transform(phase) 