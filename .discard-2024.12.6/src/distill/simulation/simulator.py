import torch
import torch.nn as nn
from config.simulation_config import (
    WAVELENGTH, SPEED_LIGHT,
    Nx, Ny, dx, simulation_step,
    ports, port_width, ports_slice,
    source_loc, detector_loc,
    center_size, center_slice,
    x_centers, y_centers, backend
)
from . import fdtd as oNN

# 设置FDTD后端
oNN.set_backend(backend)

class FDTDSimulator(nn.Module):
    """
    FDTD仿真器

    专注于执行FDTD仿真，根据给定的圆孔分布和输入端口相位，计算输出端口的响应。

    Args:
        radius_matrix (torch.Tensor): 圆孔半径矩阵，形状为 (circle_count, circle_count)
    """

    def __init__(self, radius_matrix: torch.Tensor):
        super(FDTDSimulator, self).__init__()
        
        # 处理半径矩阵
        radius_matrix = radius_matrix.clone()
        radius_matrix[radius_matrix < 0.3] = 0  # 过滤小于0.3的圆孔
        self.register_buffer('radius', (radius_matrix * 10).flatten())
        
        # 初始化仿真网格和介电常数
        self.grid = None
        self.permittivity = None

    def _initialize_grid(self):
        """初始化FDTD仿真网格，设置边界条件、波导和探测器"""
        self.grid = oNN.Grid(
            shape=(Nx, Ny, 1),
            grid_spacing=dx,
            permittivity=1.0,
        )

        # 设置PML边界条件
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

    def _set_permittivity(self):
        """根据半径矩阵设置介电常数分布"""
        if self.permittivity is None:
            # 生成网格坐标
            dtype = torch.float64 if 'float64' in backend else torch.float32
            x, y = torch.meshgrid(
                torch.arange(center_size, device=self.radius.device, dtype=dtype),
                torch.arange(center_size, device=self.radius.device, dtype=dtype),
                indexing='ij'
            )
            
            # 计算圆孔分布
            permittivity = torch.ones((center_size, center_size), device=self.radius.device, dtype=dtype)
            for idx in range(len(self.radius)):
                mask = (x - x_centers[idx]) ** 2 + (y - y_centers[idx]) ** 2 <= self.radius[idx] ** 2
                permittivity[mask] = 0  # 设置为介电常数为0的区域
            
            self.permittivity = permittivity.unsqueeze(-1) * 1.8 + 1  # 调整介电常数

            # 设置介电常数到网格
            self.grid[center_slice, center_slice, 0] = oNN.Object(
                permittivity=self.permittivity,
                name="core"
            )

    def _set_source(self, source: torch.Tensor):
        """设置输入源的相位信息

        Args:
            source (torch.Tensor): 输入端口的相位信息，形状为 (ports,)
        """
        for i in range(ports):
            self.grid[source_loc, ports_slice[i], 0] = oNN.LineSource(
                period=WAVELENGTH / SPEED_LIGHT,
                phase_shift=source[i] * torch.pi,
                name=f"source{i}"
            )

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """执行FDTD仿真

        Args:
            source (torch.Tensor): 输入端口的相位信息，形状为 (batch_size, ports)

        Returns:
            torch.Tensor: 仿真结果，形状为 (batch_size, simulation_step, ports)
        """
        batch_size = source.size(0)
        results = []

        for batch_idx in range(batch_size):
            # 初始化仿真网格
            self._initialize_grid()
            self._set_permittivity()
            self._set_source(source[batch_idx])

            # 运行仿真
            self.grid.run(simulation_step, progress_bar=False)

            # 收集探测器数据
            output = torch.empty((simulation_step, ports), device=source.device)
            for port_idx in range(ports):
                detector_values_E = torch.stack(
                    self.grid.detectors[port_idx].detector_values()["E"],
                    dim=0
                )
                detector_intensity = torch.mean(detector_values_E[:, :, -1].pow(2), dim=1)
                output[:, port_idx] = detector_intensity

            results.append(output.unsqueeze(0))  # 添加batch维度

        return torch.cat(results, dim=0)