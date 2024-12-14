import torch
import torch.nn as nn
from typing import Optional, Tuple

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
        
        # LSTM和输出层设置
        self.lstm = nn.LSTM(
            input_size=input_size + 1,  # +1 是因为输入数据已经包含了额外的通道
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.to(device)
        
    def _validate_input(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        验证输入数据格式

        Args:
            inputs: 输入数据，形状应为 (batch_size, time_steps, ports+1) 或 (time_steps, ports+1)

        Returns:
            Tuple[torch.Tensor, bool]: 验证后的输入数据和是否为批量输入的标志

        Raises:
            ValueError: 当输入数据格式不正确时
        """
        if inputs.dim() not in [2, 3]:
            raise ValueError(f"输入维度必须是2或3，当前维度为{inputs.dim()}")
        
        is_batch = inputs.dim() == 3
        if not is_batch:
            inputs = inputs.unsqueeze(0)
            
        if len(inputs.shape) != 3:
            raise ValueError(f"处理后的输入形状必须是(batch_size, time_steps, ports+1)，当前形状为{inputs.shape}")
            
        inputs = inputs.to(self.device)
        return inputs, is_batch
            
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            inputs: 输入数据，形状为 (batch_size, time_steps, ports+1) 或 (time_steps, ports+1)

        Returns:
            torch.Tensor: 预测的输出光强
        """
        # 验证输入数据
        inputs, is_batch = self._validate_input(inputs)
        
        # LSTM处理
        lstm_out, _ = self.lstm(inputs)
        final_hidden = lstm_out[:, -1, :]
        output = self.output_layer(final_hidden)
        intensity = output.pow(2)
        
        if not is_batch:
            intensity = intensity.squeeze(0)
            
        return intensity
    
    def get_sequence_output(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        获取完整的序列输出

        Args:
            inputs: 输入数据，形状为 (batch_size, time_steps, ports+1) 或 (time_steps, ports+1)

        Returns:
            torch.Tensor: 输出光强序列，形状为 (batch_size, time_steps, ports) 或 (time_steps, ports)
        """
        # 验证输入数据
        inputs, is_batch = self._validate_input(inputs)
        inputs = inputs.detach().requires_grad_(True)
        
        with torch.enable_grad():
            lstm_out, _ = self.lstm(inputs)
            output_seq = self.output_layer(lstm_out)
            intensity_seq = output_seq.pow(2)
            
            # 归一化
            time_steps = intensity_seq.size(1)
            batch_sums = intensity_seq.sum(dim=(1, 2), keepdim=True)
            intensity_seq = intensity_seq * (time_steps / (batch_sums + 1e-8))
            
            if not is_batch:
                intensity_seq = intensity_seq.squeeze(0)
            
            return intensity_seq