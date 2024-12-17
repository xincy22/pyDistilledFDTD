import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    """基于LSTM的学生模型，用于学习FDTD的输入输出映射"""
    
    def __init__(
        self,
        input_size: int,          # 输入端口数
        hidden_size: int,         # LSTM隐藏层大小
        output_size: int,         # 输出端口数
        num_layers: int = 2,      # LSTM层数    
        dropout: float = 0.1     # dropout率
    ):
        """
        初始化LSTM预测器

        Args:
            input_size: 输入端口数量
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: dropout率
        """
        super(LSTMPredictor, self).__init__()
        
        # LSTM和输出层设置
        self.lstm = nn.LSTM(
            input_size=input_size + 1,  # +1 是因为输入数据已经包含了额外的通道
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_size, output_size)
            
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回完整的时序输出

        Args:
            inputs: 输入数据，形状为 (batch_size, time_steps, ports+1) 或 (time_steps, ports+1)

        Returns:
            torch.Tensor: 输出光强序列，形状为 (batch_size, time_steps, ports) 或 (time_steps, ports)
        """

        # 验证输入数据
        if inputs.dim() not in [2, 3]:
            raise ValueError(f"输入维度必须是2或3，当前维度为{inputs.dim()}")
        
        is_batch = inputs.dim() == 3
        if not is_batch:
            inputs = inputs.unsqueeze(0)
            
        if len(inputs.shape) != 3:
            raise ValueError(
                f"处理后的输入形状必须是(batch_size, time_steps, ports+1), ",
                "当前形状为{inputs.shape}"
            )

        # 将输入转换为叶子张量，并设置requires_grad=True
        inputs = inputs.detach().requires_grad_(True)
        
        # LSTM处理
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
    
    def get_final_output(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        获取最后一个时间步的输出

        Args:
            inputs: 输入数据，形状为 (batch_size, time_steps, ports+1) 或 (time_steps, ports+1)

        Returns:
            torch.Tensor: 最后一个时间步的输出光强，形状为 (batch_size, ports) 或 (ports,)
        """
        # 验证输入数据
        # 验证输入数据
        if inputs.dim() not in [2, 3]:
            raise ValueError(f"输入维度必须是2或3，当前维度为{inputs.dim()}")
        
        is_batch = inputs.dim() == 3
        if not is_batch:
            inputs = inputs.unsqueeze(0)
            
        if len(inputs.shape) != 3:
            raise ValueError(
                f"处理后的输入形状必须是(batch_size, time_steps, ports+1), ",
                "当前形状为{inputs.shape}"
            )

        print("inputs requires_grad:", inputs.requires_grad)
        
        # LSTM处理
        lstm_out, _ = self.lstm(inputs)
        final_hidden = lstm_out[:, -1, :]  # 只取最后一个时间步
        output = self.output_layer(final_hidden)
        intensity = output.pow(2)
        
        if not is_batch:
            intensity = intensity.squeeze(0)
            
        return intensity