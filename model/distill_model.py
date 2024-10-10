import hashlib
import os
import shutil
import uuid

import numpy as np
import torch

import config as cfg
from . import fdtd as oNN

oNN.set_backend(cfg.backend)


class DistillModel(torch.nn.Module):
    """
    用于从有限差分时域（FDTD）仿真中蒸馏知识到神经网络的模型。

    该模型利用 FDTD 仿真和基于 LSTM 的学生模型进行学习。
    它包括初始化和运行 FDTD 仿真的方法，设置源，设置介电常数，并进行前向推理。
    在训练过程中，它计算 FDTD 仿真和学生模型预测之间的损失。

    :param radius_matrix: np.ndarray
        一个表示圆形半径的矩阵，形状为 `(circle_count, circle_count)`。
        用于在仿真中生成不同大小的圆形结构，所有小于 0.3 的半径会被设置为 0。
    :type radius_matrix: np.ndarray of shape (circle_count, circle_count)

    :param student_model_class: type
        一个神经网络模型类，必须是 `torch.nn.Module` 的子类。该类将用于初始化
        一个学生模型，该学生模型用于学习和模拟 FDTD 仿真输出。
    :type student_model_class: type

    :raises ValueError: 当 `student_model_class` 不是 `torch.nn.Module` 的子类时抛出错误。
    """
    def __init__(self, radius_matrix: np.ndarray, student_model_class: type):
        super(DistillModel, self).__init__()

        if not isinstance(student_model_class, type) or not issubclass(student_model_class, torch.nn.Module):
            raise ValueError("student_model_class must be a subclass of torch.nn.Module")
        radius_matrix[radius_matrix < 0.3] = 0
        self.radius = torch.tensor(radius_matrix * 10, device=cfg.device).flatten()
        self.student_model = student_model_class(cfg.ports, 128, cfg.ports).to(cfg.device)
        self.grid = None
        self.permittivity = None
        self.fdtd_simulation: bool = True
        self.lstm_simulation: bool = False

        # Create a unique cache directory for this instance
        dir_path = os.path.dirname(os.path.abspath(__file__))
        unique_id = uuid.uuid4().hex  # Unique identifier for this instance
        self.loader_cache_dir = os.path.join(dir_path, ".cache", unique_id)
        os.makedirs(self.loader_cache_dir, exist_ok=True)

    def __del__(self):
        """Destructor to clean up the cache directory when the instance is destroyed."""
        if os.path.exists(self.loader_cache_dir):
            shutil.rmtree(self.loader_cache_dir)

    def grid_init(self):
        """
        初始化 FDTD 仿真的网格。

        此方法设置网格参数，包括网格形状、网格间距和介电常数。它还为网格添加了
        完美匹配层（PML）边界，用于吸收波动并防止反射。

        此外，还会初始化用于每个端口的波导和探测器。

        :raises RuntimeError: 如果网格初始化失败，将抛出运行时错误。
        """
        self.grid = oNN.Grid(
            shape=(cfg.Nx, cfg.Ny, 1),
            grid_spacing=cfg.dx,
            permittivity=1.0
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
        设置电磁源的相位信息。

        :param source: 输入的源相位信息，每个端口对应一个相位值。
        :type source: torch.Tensor of shape (ports,)
        """
        for i in range(cfg.ports):
            self.grid[cfg.source_loc, cfg.ports_slice[i], 0] = oNN.LineSource(
                period=cfg.WAVELENGTH / cfg.SPEED_LIGHT, phase_shift=source[i] * torch.pi, name=f"source{i}")

    def set_permittivity(self):
        """
        设置仿真中的介电常数分布。

        该方法根据 `radius_matrix` 计算网格中圆形区域的介电常数分布。
        如果之前已经设置了介电常数，则直接应用之前的值。
        """
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

    def run_simulation(self, sou):
        """
        运行单个源的 FDTD 仿真。

        此方法初始化网格、设置源和介电常数，然后运行仿真，收集输出数据。

        :param sou: 单个源值，表示各个端口的源相位值。
        :type sou: torch.Tensor of shape (ports,)

        :returns:
            - output: 包含各端口总输出强度的张量，形状为 `(ports,)`。
            - output_by_time: 包含各时间步端口输出强度的张量，形状为 `(ports, time_step)`。
        :rtype: tuple(torch.Tensor of shape (ports,), torch.Tensor of shape (ports, time_step))
        """
        self.grid_init()
        self.set_source(sou)
        self.set_permittivity()
        self.grid.run(cfg.time_step, progress_bar=False)

        output = torch.tensor([], device=cfg.device)
        output_by_time = torch.tensor([], device=cfg.device)
        for i in range(cfg.ports):
            detector_values_E = torch.stack(
                self.grid.detectors[i].detector_values()["E"], dim=0
            )  # torch.Tensor[time_step, len, 3]
            detector_intensity = torch.mean(
                detector_values_E[:, :, -1] ** 2, dim=1
            )  # torch.Tensor[time_step]
            output = torch.cat(
                [output, detector_intensity.sum(dim=0).unsqueeze(0)], dim=0
            )  # torch.Tensor[ports]
            output_by_time = torch.cat(
                [output_by_time, detector_intensity.unsqueeze(0)], dim=0
            )  # torch.Tensor[ports, time_step]
        return output, output_by_time

    def sim(self, source):
        """
        运行 FDTD 仿真，并使用缓存机制避免重复计算。

        对于每个源值，检查是否已有缓存结果。如果有，则加载缓存数据；
        否则，运行仿真并保存结果到缓存。

        :param source: 输入的源值，用于每个批次中的各个端口。
        :type source: torch.Tensor of shape (batch_size, ports)

        :returns: 返回两个张量：
            - `batch_output` 包含每个批次的端口总输出强度，形状为 `(batch_size, ports)`。
            - `batch_output_by_time` 包含每个批次中各个时间步的端口输出强度，形状为 `(batch_size, ports, time_step)`。
        :rtype: tuple(torch.Tensor of shape (batch_size, ports), torch.Tensor of shape (batch_size, ports, time_step))
        """
        batch_output = []
        batch_output_by_time = []
        for sou in source:
            # Convert 'sou' to CPU for hashing
            sou_cpu = sou.cpu()
            # Compute hash of 'sou'
            sou_hash = hashlib.sha256(sou_cpu.numpy().tobytes()).hexdigest()

            # Build the cache file path using hierarchical directories
            cache_subdir = os.path.join(self.loader_cache_dir, sou_hash[:2], sou_hash[2:4])
            os.makedirs(cache_subdir, exist_ok=True)
            cache_file = os.path.join(cache_subdir, sou_hash[4:] + ".pt")

            if os.path.exists(cache_file):
                # Load the data from cache
                cached_data = torch.load(cache_file)
                # Verify that the cached 'sou' matches the current 'sou'
                if torch.equal(cached_data['source'], sou_cpu):
                    output = cached_data['output'].to(cfg.device)
                    output_by_time = cached_data['output_by_time'].to(cfg.device)
                else:
                    # Hash collision occurred, run simulation and update cache
                    output, output_by_time = self.run_simulation(sou)
                    # Move tensors to CPU to save GPU memory
                    output_cpu = output.cpu()
                    output_by_time_cpu = output_by_time.cpu()
                    # Save the data to cache with 'source'
                    torch.save({'source': sou_cpu, 'output': output_cpu, 'output_by_time': output_by_time_cpu}, cache_file)
            else:
                # Run the simulation
                output, output_by_time = self.run_simulation(sou)
                # Move tensors to CPU to save GPU memory
                output_cpu = output.cpu()
                output_by_time_cpu = output_by_time.cpu()
                # Save the data to cache with 'source'
                torch.save({'source': sou_cpu, 'output': output_cpu, 'output_by_time': output_by_time_cpu}, cache_file)

            batch_output.append(output)
            batch_output_by_time.append(output_by_time)

        # Stack the outputs
        batch_output = torch.stack(batch_output, dim=0)  # Shape: (batch_size, ports)
        batch_output_by_time = torch.stack(batch_output_by_time, dim=0)  # Shape: (batch_size, ports, time_step)
        return batch_output, batch_output_by_time

    def forward(self, source):
        """
        进行前向推理操作，使用 FDTD 仿真或学生模型，或者两者的组合。

        在评估模式下 (`self.training == False`)，该方法可以根据 `fdtd_simulation`
        和 `lstm_simulation` 标志位返回 FDTD 仿真结果、学生模型结果或两者。
        在训练模式下，该方法返回 FDTD 仿真结果与学生模型结果之间的损失。

        :param source: 输入的源值，表示每个批次中各个端口的源相位值。
        :type source: torch.Tensor of shape (batch_size, ports)

        :returns: 根据训练或评估模式返回不同结果。
            - 评估模式下返回 FDTD 仿真结果、学生模型结果或两者。
            - 训练模式下返回 FDTD 仿真结果与学生模型结果之间的损失。
        :rtype: torch.Tensor or tuple(torch.Tensor, torch.Tensor)
        """
        if not self.training:
            if self.fdtd_simulation and self.lstm_simulation:
                lstm_output = self.student_model(source)
                if lstm_output.dim() == 3 and lstm_output.size(1) == cfg.time_step:
                    lstm_output = lstm_output.sum(dim=1)
                return self.sim(source)[0], lstm_output
            if self.fdtd_simulation:
                return self.sim(source)[0]
            if self.lstm_simulation:
                lstm_output = self.student_model(source)
                if lstm_output.dim() == 3 and lstm_output.size(1) == cfg.time_step:
                    lstm_output = lstm_output.sum(dim=1)
                return lstm_output
        else:
            with torch.enable_grad():
                lstm_output = self.student_model(source)
            fdtd_output = self.sim(source)
            return self.student_model.loss(fdtd_output, lstm_output)

    def train(self, mode=True):
        """
        设置模型的训练模式。

        :param mode: 设置训练模式，如果为 `True`，则使用训练模式。
        :type mode: bool
        """
        super(DistillModel, self).train(mode)
        self.student_model.train(mode)

    def eval(self):
        """
        设置模型的评估模式。
        """
        super(DistillModel, self).eval()
        self.student_model.eval()

    def fdtd(self, mode=True):
        """
        设置 FDTD 仿真模式。

        :param mode: 设置 FDTD 仿真模式，如果为 `True`，则使用 FDTD 仿真。
        :type mode: bool
        """
        self.fdtd_simulation = mode

    def lstm(self, mode=True):
        """
        设置学生模型模式。

        :param mode: 设置学生模型模式，如果为 `True`，则使用学生模型。
        :type mode: bool
        """
        self.lstm_simulation = mode

    def set_simulation_mode(self, fdtd=True, lstm=False):
        """
        设置仿真模式。

        :param fdtd: 设置 FDTD 仿真模式，如果为 `True`，则使用 FDTD 仿真。
        :type fdtd: bool

        :param lstm: 设置学生模型模式，如果为 `True`，则使用学生模型。
        :type lstm: bool
        """
        self.fdtd(fdtd)
        self.lstm(lstm)

    def to(self, device):
        """
        将模型移动到指定的设备。

        :param device: 目标设备。
        :type device: torch.device
        """
        self.student_model.to(device)
        return super(DistillModel, self).to(device)


if __name__ == "__main__":
    from dataset import core_data_loader
    from student_model import StudentOutputModel

    train_loader, test_loader = core_data_loader(batch_size=4, eta=0.02, method="kmeans")

    print(len(train_loader.dataset))

    model = DistillModel(np.random.rand(10, 10), student_model_class=StudentOutputModel)
    model.eval()
    model.set_simulation_mode(fdtd=True, lstm=False)
    for data, labels in train_loader:
        data = data.to(cfg.device)
        labels = labels.to(cfg.device)
        print(data.dtype)
        print(model(data))
        break
