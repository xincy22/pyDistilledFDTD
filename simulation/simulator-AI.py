import os
import torch
import torch.nn as nn
import numpy as np

from . import fdtd as oNN
from .config import (
    Nx, Ny, dx, simulation_step, ports, port_width, ports_slice,
    source_loc, detector_loc, center_size, center_slice,
    circle_count, x_centers, y_centers,
    WAVELENGTH, SPEED_LIGHT,
    backend, device
)

oNN.set_backend(backend)


class FDTDSimulator(nn.Module):
    def __init__(self, radius_matrix: torch.Tensor | np.ndarray):
        super().__init__()
        self.device = device

        # to tensor
        if isinstance(radius_matrix, np.ndarray):
            radius_matrix = torch.tensor(radius_matrix, dtype=torch.float64, device=device)
        
        radius_matrix[radius_matrix < 0.3] = 0

        # flatten
        self.radius_matrix = radius_matrix.flatten()
        
        # 初始化网格
        self.grid = None
        self.permittivity = None

    def _init_grid(self):
        """初始化网格，设置PML边界和波导"""
        self.grid = oNN.Grid(
            shape=(Nx, Ny, 1),
            grid_spacing=dx,
            permittivity=1.0,
        )

        # 设置PML边界
        self.grid[0:10, :, :] = oNN.PML(name="pml_xlow")
        self.grid[-10:, :, :] = oNN.PML(name="pml_xhigh")
        self.grid[:, 0:10, :] = oNN.PML(name="pml_ylow")
        self.grid[:, -10:, :] = oNN.PML(name="pml_yhigh")

        # 设置波导和探测器
        for i in range(ports):
            # 输入波导
            self.grid[0:source_loc, ports_slice[i], 0] = oNN.Object(
                permittivity=2.8 * torch.ones([source_loc, port_width, 1]),
                name=f"wg{i}"
            )
            # 输出波导
            self.grid[detector_loc:, ports_slice[i], 0] = oNN.Object(
                permittivity=2.8 * torch.ones([Nx - detector_loc, port_width, 1]),
                name=f"op{i}"
            )
            # 探测器
            self.grid[detector_loc, ports_slice[i], 0] = oNN.LineDetector(
                name=f"detector{i}"
            )

    def _set_source(self, source: torch.Tensor):
        """
        设置源的相位

        Args:
            source (torch.Tensor): 输入数据，作为输入源的相位进行编码，形状为(ports,)
        """
        for i in range(ports):
            self.grid[source_loc, ports_slice[i], 0] = oNN.LineSource(
                period=WAVELENGTH / SPEED_LIGHT,
                phase_shift=source[i] * torch.pi,
                name=f"source{i}"
            )

    def _set_permittivity(self):
        """根据半径矩阵设置介电常数"""
        if self.permittivity is not None:
            self.grid[center_slice, center_slice, 0] = oNN.Object(
                permittivity=self.permittivity * 1.8 + 1,
                name="permittivity"
            )
            return

        # 计算圆形区域
        x, y = torch.meshgrid(
            torch.arange(center_size, device=self.device),
            torch.arange(center_size, device=self.device),
            indexing='ij'
        )

        outside_circle = torch.ones(
            (center_size, center_size),
            dtype=torch.int,
            device=self.device
        )

        # 设置每个圆形区域
        for i in range(circle_count * circle_count):
            mask = (x - x_centers[i]) ** 2 + (y - y_centers[i]) ** 2 <= self.radius_matrix[i] ** 2
            outside_circle[mask] = 0

        # 转换数据类型
        if torch.cuda.is_available():
            self.permittivity = outside_circle.view(
                center_size, center_size, 1
            ).double()
        else:
            self.permittivity = outside_circle.view(
                center_size, center_size, 1
            ).float()

        # 设置到网格中
        self.grid[center_slice, center_slice, 0] = oNN.Object(
            permittivity=self.permittivity * 1.8 + 1,
            name="core"
        )

    def forward(self, input: torch.Tensor):
        """
        进行FDTD仿真

        Args:
            input (torch.Tensor): 输入数据，形状为(batch_size, ports)或(ports,)

        Returns:
            torch.Tensor: 仿真结果，形状为(batch_size, ports)或(ports,)
        """
        # 确保输入是2D张量
        if input.dim() == 1:
            input = input.unsqueeze(0)
        
        # 初始化网格
        self._init_grid()
        
        # 设置介电常数
        self._set_permittivity()
        
        # 对每个batch进行仿真
        results = []
        for source in input:
            # 设置源
            self._set_source(source)
            
            # 运行仿真
            for _ in range(simulation_step):
                self.grid.step()
            
            # 获取探测器数据
            detector_data = []
            for i in range(ports):
                detector = self.grid[detector_loc, ports_slice[i], 0]
                # 取最后一个时间步的数据
                detector_data.append(detector.data[-1])
            
            results.append(torch.stack(detector_data))
        
        # 堆叠所有batch的结果
        output = torch.stack(results)
        
        # 如果输入是1D的，返回1D结果
        if input.size(0) == 1:
            output = output.squeeze(0)
        
        return output

