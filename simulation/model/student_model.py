import torch

class StudentLSTMModule(torch.nn.Module):
    """
    LSTM模型

    :param input_size: 输入的特征维度。
    :type input_size: int

    :param hidden_size: LSTM 层的隐藏单元数。
    :type hidden_size: int

    :param output_size: 输出的特征维度。
    :type output_size: int

    :param num_layer: LSTM 的层数，默认为 2。
    :type num_layer: int

    :ivar lstm: 用于处理时间步长序列的 LSTM 层。
    :type lstm: torch.nn.LSTM

    :ivar fc: 用于将 LSTM 输出映射到输出空间的全连接层。
    :type fc: torch.nn.Linear
    """
    def __init__(self, input_size, hidden_size, output_size, num_layer=2):
        super(StudentLSTMModule, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layer, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x) # lstm_out: (batch_size, seq_len, hidden_size)
        return self.fc(lstm_out) # (batch_size, seq_len, output_size)