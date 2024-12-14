import torch
import torch.nn as nn
import torch.nn.functional as F  # 在文件开头添加这行
from typing import Optional, Tuple
from ..config import WAVELENGTH, SPEED_LIGHT, simulation_step
from .signal_expansion import SignalExpansion

class LSTMPredictor(nn.Module):
    """基于LSTM的学生模型，用于学习FDTD的输入输出映射"""
    
    def __init__(
        self,
        input_size: int,          # 输入端口数
        hidden_size: int,         # LSTM隐藏层大小
        output_size: int,         # 输出端口数
        num_layers: int = 2,      # LSTM层数    
        dropout: float = 0.1,     # dropout率
        device: Optional[torch.device] = None
    ):
        """
        初始化LSTM预测器

        Args:
            input_size: 输入端口数量
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: dropout率
            device: 计算设备
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device if device is not None else torch.device('cpu')
        
        # 创建信号扩展器，使用配置中的参数
        self.signal_expander = SignalExpansion(
            wavelength=WAVELENGTH,
            speed_light=SPEED_LIGHT,
            time_steps=simulation_step,
            device=self.device
        )
        
        # 确保LSTM的参数有梯度
        self.lstm = nn.LSTM(
            input_size=input_size + 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 确保输出层的参数有梯度
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # 将模型移动到指定设备并确保参数可训练
        self.to(device)
        for param in self.parameters():
            param.requires_grad = True
        
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
        
    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            phase: 输入相位，形状为 (batch_size, ports) 或 (ports,)

        Returns:
            torch.Tensor: 预测的输出光强
        """
        # 记录输入维度
        is_batch = phase.dim() == 2
        if not is_batch:
            phase = phase.unsqueeze(0)
            
        # 将相位扩展为时序信号
        signal = self.signal_expander(phase)  # (batch_size, time_steps, ports)
        
        # 添加零值通道
        signal = self._add_zero_channel(signal)  # (batch_size, time_steps, ports+1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(signal)  # (batch_size, time_steps, hidden_size)
        
        # 只使用最后一个时间步的输出
        final_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # 生成输出场强
        output = self.output_layer(final_hidden)  # (batch_size, ports)
        
        # 转换为光强
        intensity = output.pow(2)
        
        # 如果输入不是批量的，去掉批量维度
        if not is_batch:
            intensity = intensity.squeeze(0)
            
        return intensity
    
    def get_sequence_output(self, phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取完整的序列输出，用于分析模型的时序行为

        Args:
            phase: 输入相位，形状为 (batch_size, ports) 或 (ports,)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - 输入序列（带零值通道），形状为 (batch_size, time_steps, ports+1)
                - 输出光强序列，形状为 (batch_size, time_steps, ports)
        """
        # 记录输入维度
        is_batch = phase.dim() == 2
        if not is_batch:
            phase = phase.unsqueeze(0)
        
        # 生成输入序列 - 不需要梯度
        with torch.no_grad():
            input_seq = self.signal_expander(phase)
            input_seq = self._add_zero_channel(input_seq)
        
        # 从这里开始计算梯度
        input_seq = input_seq.detach().requires_grad_(True)

        with torch.enable_grad():
            lstm_out, _ = self.lstm(input_seq)
        
            # 生成输出序列
            output_seq = self.output_layer(lstm_out)
            
            # 转换为光强序列
            intensity_seq = output_seq.pow(2)
            
            # 归一化：让每个batch的所有输出的和等于时间步数
            time_steps = intensity_seq.size(1)  # 获取时间步数
            # 对每个batch单独计算和并归一化
            batch_sums = intensity_seq.sum(dim=(1, 2), keepdim=True)  # 计算每个batch的总和
            intensity_seq = intensity_seq * (time_steps / (batch_sums + 1e-8))
                
            if not is_batch:
                input_seq = input_seq.squeeze(0)
                intensity_seq = intensity_seq.squeeze(0)
            
            return input_seq, intensity_seq