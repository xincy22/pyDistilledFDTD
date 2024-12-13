import torch
import numpy as np
from typing import Optional

class SignalExpansion:
    """将相位信号扩展为时序信号的类"""
    
    def __init__(
        self,
        wavelength: float = 1.0,
        speed_light: float = 3e8,
        time_steps: int = 1000,
        device: Optional[torch.device] = None
    ):
        """
        初始化信号扩展器

        Args:
            wavelength: 波长
            speed_light: 光速
            time_steps: 时间步数
            device: 计算设备
        """
        self.wavelength = wavelength
        self.speed_light = speed_light
        self.time_steps = time_steps
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