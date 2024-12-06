import os
import uuid
import hashlib
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from . import fdtd as oNN
from . import config as cfg

oNN.set_backend("torch.cuda.float64" if torch.cuda.is_available() else "torch.float64")

class DistillModel(torch.nn.Module):
    """
    一个用于知识蒸馏的模型，用于对 FDTD 模型进行知识蒸馏。

    :param radius_matrix: 一个numpy数组，表示每个圆形区域的半径。
    :type radius_matrix: np.ndarray

    :param student_model: 用于知识蒸馏的学生模型。
    :type student_model: torch.nn.Module

    :param expand_method: 数据扩展方法，默认为 "repeat"
    :type expand_method: str

    :ivar radius: 一个张量，表示每个圆形区域的半径。
    :type radius: torch.Tensor

    :ivar student_model: 用于知识蒸馏的学生模型。
    :type student_model: torch.nn.Module

    :ivar expand_method: 数据扩展方法，默认为 "repeat"
    :type expand_method: str

    :ivar criterion: 用于计算损失的损失函数。
    :type criterion: torch.nn.MSELoss

    :ivar grid: 用于 FDTD 仿真的网格。
    :type grid: oNN.Grid

    :ivar permittivity: 介电常数。
    :type permittivity: torch.Tensor

    :ivar fdtd_simulation: 是否使用 FDTD 仿真。
    :type fdtd_simulation: bool

    :ivar lstm_simulation: 是否使用 LSTM 模拟。
    :type lstm_simulation: bool

    :ivar unique_id: 唯一标识符。
    :type unique_id: str

    :ivar cache_dir: 缓存目录。
    :type cache_dir: str

    Methods
    -------
    - `train` : 设置模型为训练模式
    - `eval` : 设置模型为评估模式
    - `fdtd` : 是否使用 FDTD 仿真
    - `lstm` : 是否使用 LSTM 模拟
    - `to` : 将模型转移到指定设备
    - `grid_init` : 初始化网格
    - `set_source` : 设置源的相位
    - `set_permittivity` : 根据半径矩阵设置介电常数
    """
    def __init__(
            self, radius_matrix : np.ndarray,
            student_model : torch.nn.Module,
            expand_method : str = "repeat"
    ):
        super(DistillModel, self).__init__()

        radius_matrix[radius_matrix < 0.3] = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.radius = torch.tensor(radius_matrix * 10, device=self.device).flatten()
        self.student_model = student_model
        self.expand_method = expand_method
        self.criterion : torch.nn.MSELoss = torch.nn.MSELoss()

        self.grid = oNN.Grid(
            shape=(cfg.Nx, cfg.Ny, 1),
            grid_spacing=cfg.dx,
            permittivity=1.0,
        )
        self.permittivity = None
        self.fdtd_simulation : bool = True
        self.lstm_simulation : bool = False

        self.unique_id = uuid.uuid4().hex
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.cache_dir = os.path.join(dir_path, ".cache", self.unique_id)
        os.makedirs(self.cache_dir, exist_ok=True)

    def __del__(self):
        """删除缓存目录"""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def train(self, mode=True):
        """
        调用该函数会开启或关闭训练模式。
        关闭训练模式，自动进入评估模式，设置仿真模式以退出评估模式。

        :param mode: 是否为训练模式，默认为 True
        :type mode: bool
        """
        super(DistillModel, self).train(mode)
        self.student_model.train(mode)
        self.fdtd_simulation = False
        self.lstm_simulation = False

    def eval(self):
        """
        设置模型为评估模式

        评估模式下，fdtd_simulation, lstm_simulation都会被设置为False
        """
        self.train(False)
        self.fdtd_simulation = False
        self.lstm_simulation = False

    def fdtd(self, mode=True):
        """是否使用 FDTD 仿真"""
        self.fdtd_simulation = mode
        self.lstm_simulation = not mode

    def lstm(self, mode=True):
        """是否使用 LSTM 模拟"""
        self.lstm_simulation = mode
        self.fdtd_simulation = not mode

    def to(self, device):
        """
        将模型转移到指定设备

        :param device: 设备
        :type device: torch.device or str
        :return: self
        """
        self.radius = self.radius.to(device)
        if self.permittivity is not None:
            self.permittivity = self.permittivity.to(device)
        self.student_model = self.student_model.to(device)
        return super(DistillModel, self).to(device)

    def grid_init(self):
        """初始化网格"""
        self.grid = oNN.Grid(
            shape=(cfg.Nx, cfg.Ny, 1),
            grid_spacing=cfg.dx,
            permittivity=1.0,
        )

        self.grid[0:10, :, :] = oNN.PML(name="pml_xlow")
        self.grid[-10:, :, :] = oNN.PML(name="pml_xhigh")
        self.grid[:, 0:10, :] = oNN.PML(name="pml_ylow")
        self.grid[:, -10:, :] = oNN.PML(name="pml_yhigh")

        for i in range(cfg.ports):
            self.grid[0:cfg.source_loc, cfg.ports_slice[i], 0] = oNN.Object(
                permittivity=2.8 * torch.ones([cfg.source_loc, cfg.port_width, 1]), name=f"wg{i}")
            self.grid[cfg.detector_loc:, cfg.ports_slice[i], 0] = oNN.Object(
                permittivity=2.8 * torch.ones([cfg.Nx - cfg.detector_loc, cfg.port_width, 1]), name=f"op{i}")
            self.grid[cfg.detector_loc, cfg.ports_slice[i], 0] = oNN.LineDetector(
                name=f"detector{i}")

    def set_source(self, source):
        """
        设置源的相位

        :param source: 输入数据，作为输入源的相位进行编码
        :type source: torch.Tensor of shape (ports,)
        """
        for i in range(cfg.ports):
            self.grid[cfg.source_loc, cfg.ports_slice[i], 0] = oNN.LineSource(
                period=cfg.WAVELENGTH / cfg.SPEED_LIGHT, phase_shift=source[i] * torch.pi, name=f"source{i}")

    def set_permittivity(self):
        """根据半径矩阵设置介电常数"""
        if self.permittivity is not None:
            self.grid[cfg.center_slice, cfg.center_slice, 0] = oNN.Object(
                permittivity=self.permittivity * 1.8 + 1, name="permittivity")
            return

        x, y = torch.meshgrid(
            torch.arange(cfg.center_size, device=cfg.device), torch.arange(cfg.center_size, device=cfg.device),
            indexing='ij'
        )

        outside_circle = torch.ones((cfg.center_size, cfg.center_size), dtype=torch.int, device=cfg.device)

        for i in range(cfg.circle_count * cfg.circle_count):
            mask = (x - cfg.x_centers[i]) ** 2 + (y - cfg.y_centers[i]) ** 2 <= self.radius[i] ** 2
            outside_circle[mask] = 0
        if cfg.backend == "torch.cuda.float64" or cfg.backend == "torch.float64":
            self.permittivity = outside_circle.view(cfg.center_size, cfg.center_size, 1).double()
        else:
            self.permittivity = outside_circle.view(cfg.center_size, cfg.center_size, 1).float()
        self.grid[cfg.center_slice, cfg.center_slice, 0] = oNN.Object(
            permittivity=self.permittivity * 1.8 + 1, name="core"
        )

    def data_expand(self, source):
        """
        将数据扩展到整个仿真步长

        根据指定的方法进行扩展，支持三种方法：
        - repeat: 重复数据
        - gaussian: 通过高斯分布曲线进行扩展
        - sin: 通过正弦波进行编码，将数据编码在相位上

        :param source: 输入数据
        :type source: torch.Tensor of shape (batch_size, ports)

        :return: 扩展后的数据
        :rtype: torch.Tensor of shape (batch_size, simulation_step, ports)
        """
        if self.expand_method == "repeat":
            return source.unsqueeze(1).expand(-1, cfg.simulation_step, -1)
        elif self.expand_method == "gaussian":
            mean = (cfg.simulation_step - 1) / 2.0
            std_dev = (cfg.simulation_step - 1) / 6.0

            t = torch.arange(cfg.simulation_step, device=self.device, dtype=source.dtype)

            gaussian_curve = torch.exp(-((t - mean) ** 2) / (2 * std_dev ** 2))
            gaussian_curve = gaussian_curve / gaussian_curve.sum()
            expanded_value = gaussian_curve.view(1, -1, 1) * source.unsqueeze(1)
            return expanded_value
        elif self.expand_method == "sin":
            period = cfg.WAVELENGTH / cfg.SPEED_LIGHT
            omega = 2 * torch.pi / period

            t = torch.arange(cfg.simulation_step, device=self.device).view(1, cfg.simulation_step, 1) * self.grid.time_step
            phase_shift = source.unsqueeze(1).to(self.device) * torch.pi
            sin_wave = torch.sin(omega * t + phase_shift)
            return sin_wave

    def forward(self, source):
        """
        前向传播函数

        在训练模式下，返回FDTD仿真和LSTM模拟的损失
        在评估模式下，根据选择的模式返回对应的输出

        :param source: 输入数据，作为输入源的相位进行编码，每个元素都在[0, 1]之间
        :type source: torch.Tensor of shape (batch_size, ports)

        :return: 
            - 训练模式 (self.training=True):
                - 返回损失值。
                - torch.Tensor of shape ()
            
            - 评估模式 (self.training=False):
                - 如果启用了FDTD仿真 (self.fdtd_simulation=True):
                    - 返回归一化后的FDTD仿真输出。
                    - torch.Tensor of shape (batch_size, ports)
                
                - 如果启用了LSTM模拟 (self.lstm_simulation=True):
                    - 返回归一化后的LSTM模拟输出。
                    - torch.Tensor of shape (batch_size, ports)
                
                - 如果未启用任何仿真:
                    - 返回FDTD仿真输出和LSTM模拟输出的元组。
                    - Tuple[torch.Tensor, torch.Tensor]
                        - 第一个张量: (batch_size, simulation_steps, ports)
                        - 第二个张量: (batch_size, simulation_steps, ports)
        :rtype: torch.Tensor 或 Tuple[torch.Tensor, torch.Tensor]
        """
        if self.training:
            with torch.no_grad():
                fdtd_output = self.sim(source)
            with torch.enable_grad():
                lstm_output = self.student_model(self.data_expand(source))
            return self.criterion(fdtd_output, lstm_output)
        else:
            if self.fdtd_simulation:
                return F.normalize(self.sim(source).sum(dim=1), dim=1)
            elif self.lstm_simulation:
                return F.normalize(self.student_model(self.data_expand(source)).sum(dim=1), dim=1)
            else:
                return self.sim(source), self.student_model(self.data_expand(source))


    def sim(self, source):
        """
        FDTD仿真
        :param source: 输入数据，作为输入源的相位进行编码
        :type source: torch.Tensor of shape (batch_size, ports)

        :return: FDTD仿真输出
        :rtype: torch.Tensor of shape (batch_size, simulation_step, ports)
        """
        batch_output = torch.tensor([], device=self.device)
        for i in range(source.shape[0]):
            source_cpu = source[i].cpu()
            source_hash = hashlib.sha256(source_cpu.numpy().tobytes()).hexdigest()
            cache_subdir = os.path.join(self.cache_dir, source_hash[:1], source_hash[1:2])
            os.makedirs(cache_subdir, exist_ok=True)
            cache_file = os.path.join(cache_subdir, f"{source_hash[2:]}.pt")

            if os.path.exists(cache_file):
                cached_data = torch.load(cache_file, weights_only=False)
                if torch.equal(cached_data['source'], source_cpu):
                    batch_output = torch.cat((
                        batch_output, cached_data['output'].to(self.device).unsqueeze(0)
                    ), dim=0)
                    continue

            self.grid_init()
            self.set_source(source[i])
            self.set_permittivity()
            self.grid.run(cfg.simulation_step, progress_bar=False)
            output = torch.tensor([], device=self.device)
            for j in range(cfg.ports):
                # torch.Tensor[simulation_step, len, 3]
                detector_values_E = torch.stack(self.grid.detectors[j].detector_values()["E"], dim=0)

                # torch.Tensor[simulation_step]
                detector_intensity = torch.mean(detector_values_E[:, :, -1].pow(2), dim=1)

                # torch.Tensor[simulation_step, ports]
                output = torch.cat((output, detector_intensity.unsqueeze(1)), dim=1)

            # torch.Tensor[batch_size, simulation_step, ports]
            batch_output = torch.cat((batch_output, output.unsqueeze(0)), dim=0)
            torch.save({"source": source_cpu, "output": output.cpu()}, cache_file)

        return batch_output

