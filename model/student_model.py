import torch
import config as cfg

def data_expand(data, time_step=cfg.time_step, method="repeat"):
    """
    将数据扩展到指定的时间步长。

    根据提供的方法对输入数据进行扩展，支持三种方法：
    - `repeat`: 重复数据，扩展为时间步长长度。
    - `gaussian`: 基于高斯分布曲线进行扩展。
    - `sin`: 基于正弦函数对数据进行扩展，生成一个时间序列，其中原始数据作为正弦函数的初始相位。

    :param data: 输入数据，形状为 `(batch_size, ports)`。
    :type data: torch.Tensor of shape (batch_size, ports)

    :param time_step: 要扩展的时间步长。
    :type time_step: int

    :param method: 扩展数据的方法，支持 `repeat`、`gaussian` 和 `sin`，默认 `repeat`。
    :type method: str

    :returns: 扩展后的数据，形状为 `(batch_size, time_step, ports)`。
    :rtype: torch.Tensor of shape (batch_size, time_step, ports)

    :raises ValueError: 当 `method` 参数不是 `repeat`、`gaussian` 或 `sin` 时，抛出该错误。
    """
    if method == "repeat":
        return data.unsqueeze(1).expand(-1, time_step, -1)
    elif method == "gaussian":
        mean = (time_step - 1) / 2.0
        std_dev = (time_step - 1) / 6.0

        t = torch.arange(time_step, device=data.device, dtype=data.dtype)

        gaussian_curve = torch.exp(-((t - mean) ** 2) / (2 * std_dev ** 2))
        gaussian_curve = gaussian_curve / gaussian_curve.sum()  # Shape: (time_step,)
        expanded_value = gaussian_curve.view(1, -1, 1) * data.unsqueeze(1)  # Shape: (batch_size, time_step, ports)
        return expanded_value
    elif method == "sin":
        from config import WAVELENGTH, SPEED_LIGHT

        period = WAVELENGTH / SPEED_LIGHT  # Calculate the period
        omega = 2 * torch.pi / period      # Angular frequency

        t = torch.arange(time_step, device=data.device, dtype=data.dtype).view(1, time_step, 1)  # Shape: (1, time_step, 1)
        phase_shift = data * torch.pi  # Scale data to range [0, π], Shape: (batch_size, ports)
        phase_shift = phase_shift.unsqueeze(1)  # Shape: (batch_size, 1, ports)

        sin_wave = torch.sin(omega * t + phase_shift)  # Shape: (batch_size, time_step, ports)
        return sin_wave
    else:
        raise ValueError("Invalid method")

class StudentOutputModel(torch.nn.Module):
    """
    一个 LSTM 基于输出模型的神经网络，处理输入数据并生成输出。

    该模型接收时间步长扩展后的输入数据，并通过 LSTM 和全连接层进行处理。

    :ivar lstm: 用于处理时间步长序列的 LSTM 层。
    :type lstm: torch.nn.LSTM

    :ivar fc: 用于将 LSTM 输出映射到输出空间的全连接层。
    :type fc: torch.nn.Linear

    :ivar criterion: 用于计算 LSTM 预测输出和目标之间的损失的损失函数。
    :type criterion: torch.nn.MSELoss
    """
    def __init__(self, input_size, hidden_size, output_size, num_layer=2):
        """
        :param input_size: 输入的特征维度。
        :type input_size: int

        :param hidden_size: LSTM 层的隐藏单元数。
        :type hidden_size: int

        :param output_size: 输出的特征维度。
        :type output_size: int

        :param num_layer: LSTM 的层数，默认为 2。
        :type num_layer: int
        """
        super(StudentOutputModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layer, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        """
        前向传播函数。

        该方法接收输入数据 `x`，将其扩展到指定的时间步长后，通过 LSTM 层进行处理，并将 LSTM 输出的时间步长
        序列按时间维度求和，最后通过全连接层生成最终的输出。

        :param x: 输入数据，形状为 `(batch_size, ports)`，表示每个批次的端口输入。
        :type x: torch.Tensor of shape (batch_size, ports)

        :returns: 处理后的输出，形状为 `(batch_size, output_size)`，表示每个批次的输出特征。
        :rtype: torch.Tensor of shape (batch_size, output_size)
        """
        out, _ = self.lstm(data_expand(x, method="sin")) # out: torch.Tensor[batch_size, time_step, hidden_size]
        out = out.sum(dim=1) # 随时间求和 out: torch.Tensor[batch_size, hidden_size]
        return self.fc(out) # torch.Tensor[batch_size, output_size]

    def loss(self, fdtd_output, lstm_output):
        """
        计算模型的损失函数。

        该方法将 FDTD 仿真输出与 LSTM 模型的预测输出进行对比，计算它们之间的均方误差 (MSE)。FDTD 仿真输出包括
        两部分：端口的总强度和按时间步长输出的强度。

        :param fdtd_output: FDTD 仿真的输出，包含两个张量：
            - 第一个张量，形状为 `(batch_size, ports)`，表示端口的总输出强度。
            - 第二个张量，形状为 `(batch_size, ports, time_step)`，表示每个时间步长的端口强度。
        :type fdtd_output: tuple(torch.Tensor of shape (batch_size, ports), torch.Tensor of shape (batch_size, ports, time_step))

        :param lstm_output: LSTM 模型的输出，形状为 `(batch_size, ports)`，表示每个批次的端口预测输出。
        :type lstm_output: torch.Tensor of shape (batch_size, ports)

        :returns: 计算得到的损失值。
        :rtype: torch.Tensor
        """
        return self.criterion(lstm_output, fdtd_output[0].detach())

class StudentSequenceModel(torch.nn.Module):
    """
    一个基于 LSTM 的序列模型，处理输入数据并生成时间步长序列的输出。

    该模型接收时间步长扩展后的输入数据，并通过 LSTM 和全连接层进行处理。

    :ivar lstm: 用于处理时间步长序列的 LSTM 层。
    :type lstm: torch.nn.LSTM

    :ivar fc: 用于将 LSTM 输出映射到输出空间的全连接层。
    :type fc: torch.nn.Linear

    :ivar criterion: 用于计算 LSTM 预测输出和目标之间的损失的损失函数。
    :type criterion: torch.nn.MSELoss
    """
    def __init__(self, input_size, hidden_size, output_size, num_layer=2):
        """
        初始化 StudentSequenceModel 类。

        :param input_size: 输入的特征维度。
        :type input_size: int

        :param hidden_size: LSTM 层的隐藏单元数。
        :type hidden_size: int

        :param output_size: 输出的特征维度。
        :type output_size: int

        :param num_layer: LSTM 的层数，默认为 2。
        :type num_layer: int
        """
        super(StudentSequenceModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layer, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        """
        前向传播。

        该方法接收输入的源数据，并通过 LSTM 处理后返回时间步长序列的输出。

        :param x: 输入的源数据，形状为 `(batch_size, ports)`。
        :type x: torch.Tensor of shape (batch_size, ports)

        :returns: 处理后的输出，形状为 `(batch_size, time_step, ports)`。
        :rtype: torch.Tensor of shape (batch_size, time_step, ports)
        """
        out, _ = self.lstm(data_expand(x, method="sin")) # out: torch.Tensor[batch_size, time_step, hidden_size]
        return self.fc(out) # torch.Tensor[batch_size, time_step, output_size]

    def loss(self, fdtd_output, lstm_output):
        """
        计算损失函数。

        该方法将 FDTD 仿真输出与 LSTM 模型的时间步长输出进行对比，计算均方误差。

        :param fdtd_output: FDTD 仿真输出，包含两个部分：
            - 第一个张量，形状为 `(batch_size, ports)`，表示仿真输出的端口总强度。
            - 第二个张量，形状为 `(batch_size, ports, time_step)`，表示仿真过程中每个时间步长的端口强度。
        :type fdtd_output: tuple(torch.Tensor of shape (batch_size, ports), torch.Tensor of shape (batch_size, ports, time_step))

        :param lstm_output: LSTM 模型的时间步长输出，形状为 `(batch_size, time_step, ports)`。
        :type lstm_output: torch.Tensor of shape (batch_size, time_step, ports)

        :returns: 损失值。
        :rtype: torch.Tensor
        """
        return self.criterion(lstm_output, fdtd_output[1].permute(0, 2, 1).detach())