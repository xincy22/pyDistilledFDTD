import numpy as np
import torch
import torch.nn as nn

from . import fdtd as oNN
from .config import (
    WAVELENGTH, SPEED_LIGHT,
    backend, device,

    Nx, Ny, dx, simulation_step,
    ports, port_width, ports_slice,
    source_loc, detector_loc,

    center_size, center_slice,
    circle_count,
    x_centers, y_centers,
)

oNN.set_backend(backend)


class FDTDSimulator(nn.Module):
    def __init__(
            self,
            radius_matrix: torch.Tensor | np.ndarray,
    ):
        super(FDTDSimulator, self).__init__()
        
        self.device = device
        
        # 转换为tensor并处理小值
        if isinstance(radius_matrix, np.ndarray):
            radius_matrix = torch.as_tensor(radius_matrix.flatten(), dtype=torch.float64, device=device)
        else:
            radius_matrix = radius_matrix.flatten().to(device)
        self.radius_matrix = radius_matrix * (radius_matrix >= 0.3)

        # 定义网格和介电常数
        self.grid = None
        self.permittivity = None

    def forward(self, inputs: torch.Tensor):
        # 检查输入维度
        if inputs.dim() == 1 and inputs.shape[0] == ports:
            inputs = inputs.unsqueeze(0)
        elif not (inputs.dim() == 2 and inputs.shape[1] == ports):
            raise ValueError(f"输入维度必须为 (ports={ports},) 或 (batch_size, ports={ports})")
        
        batch_output = []
        for i in range(inputs.shape[0]):
            batch_output.append(self.sim(inputs[i]))
        return torch.stack(batch_output, dim=0)

    def sim(self, x: torch.Tensor):
        self._init_grid()
        self._set_source(x)
        self._set_permittivity()
        self.grid.run(simulation_step, progress_bar=False)

        intensities = []
        for i in range(ports):
            detector_values_E = torch.stack(self.grid.detectors[i].detector_values()["E"], dim=0)
            detector_intensity = torch.mean(detector_values_E[:, :, -1].pow(2), dim=1)
            intensities.append(detector_intensity)

        output = torch.stack(intensities, dim=1)
        assert output.shape == (simulation_step, ports)
        
        # 归一化：让所有输出的和等于时间步数
        time_steps = output.size(0)  # 获取时间步数
        total_sum = output.sum()  # 计算总和
        output = output * (time_steps / (total_sum + 1e-8))
        
        return output

    def _init_grid(self):
        self.grid = oNN.Grid(
            shape=(Nx, Ny, 1),
            grid_spacing=dx,
            permittivity=1.0,
        )

        self.grid[0:10, :, :] = oNN.PML(name="pml_xlow")
        self.grid[-10:, :, :] = oNN.PML(name="pml_xhigh")
        self.grid[:, 0:10, :] = oNN.PML(name="pml_ylow")
        self.grid[:, -10:, :] = oNN.PML(name="pml_yhigh")

        for i in range(ports):
            self.grid[0:source_loc, ports_slice[i], 0] = oNN.Object(
                permittivity=2.8 * torch.ones([source_loc, port_width, 1]), name=f"wg{i}")
            self.grid[detector_loc:, ports_slice[i], 0] = oNN.Object(
                permittivity=2.8 * torch.ones([Nx - detector_loc, port_width, 1]), name=f"op{i}")
            self.grid[detector_loc, ports_slice[i], 0] = oNN.LineDetector(
                name=f"detector{i}")

    def _set_source(self, source: torch.Tensor):
        for i in range(ports):
            self.grid[source_loc, ports_slice[i], 0] = oNN.LineSource(
                period=WAVELENGTH / SPEED_LIGHT, phase_shift=source[i] * torch.pi, name=f"source{i}")
            
    def _set_permittivity(self):
        if self.permittivity is not None:
            self.grid[center_slice, center_slice, 0] = oNN.Object(
                permittivity=self.permittivity * 1.8 + 1, name="permittivity")
            return

        x, y = torch.meshgrid(
            torch.arange(center_size, device=device), torch.arange(center_size, device=device),
            indexing='ij'
        )   

        outside_circle = torch.ones((center_size, center_size), dtype=torch.int, device=device)

        for i in range(circle_count * circle_count):
            mask = (x - x_centers[i]) ** 2 + (y - y_centers[i]) ** 2 <= self.radius_matrix[i] ** 2
            outside_circle[mask] = 0
        
        self.permittivity = outside_circle.view(center_size, center_size, 1)
        self.grid[center_slice, center_slice, 0] = oNN.Object(
            permittivity=self.permittivity * 1.8 + 1, name="core"
        )

