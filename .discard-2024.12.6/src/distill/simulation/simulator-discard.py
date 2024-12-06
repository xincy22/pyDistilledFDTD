from config import simulation_config as cfg

from . import fdtd as oNN

import torch
import os

oNN.set_backend(cfg.backend)

class FDTDSimulator:

    def __init__(self, radius_matrix: torch.Tensor):
        """初始化FDTD模拟器"""
        radius_matrix[radius_matrix < 0.3] = 0
        self.device = cfg.device
        self.radius = (radius_matrix * 10).flatten().to(self.device)

        self.grid = None

        self.permittivity = None

    def _grid_init(self):
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

    def _set_permittivity(self):
        if self.permittivity is None:
            dtype = torch.float64 if cfg.backend.endswith("float64") else torch.float32
            x, y = torch.meshgrid(
                torch.arange(cfg.center_size, device=self.device, dtype=dtype),
                torch.arange(cfg.center_size, device=self.device, dtype=dtype),
                indexing="ij"
            )
            outside_circle = torch.ones(
                (cfg.center_size, cfg.center_size), device=self.device, dtype=dtype)
            for i in range(cfg.circle_count * cfg.circle_count):
                mask = (x - cfg.x_centers[i]) ** 2 + (y - cfg.y_centers[i]) ** 2 <= self.radius[i] ** 2
                outside_circle[mask] = 0
            self.permittivity = outside_circle.view(cfg.center_size, cfg.center_size, 1)
        self.grid[cfg.center_slice, cfg.center_slice, 0] = oNN.Object(
            permittivity=self.permittivity * 1.8 + 1, name="core"
        )

    def _set_source(self, source: torch.Tensor):
        for i in range(cfg.ports):
            self.grid[cfg.source_loc, cfg.ports_slice[i], 0] = oNN.LineSource(
                period=cfg.WAVELENGTH / cfg.SPEED_LIGHT, 
                phase_shift=source[i] * torch.pi, 
                name=f"source{i}"
            )

    def sim(self, source: torch.Tensor):
        """
        :param source: 输入数据，作为输入源的相位进行编码
        :type source: torch.Tensor of shape (batch_size, ports)

        :return: FDTD仿真输出
        :rtype: torch.Tensor of shape (batch_size, simulation_step, ports)
        """
        batch_output = torch.tensor([], device=self.device)
        for i in range(source.shape[0]):
            self._grid_init()
            self._set_permittivity()
            self._set_source(source[i])
            self.grid.run(cfg.simulation_step, progress_bar=False)
            output = torch.tensor([], device=self.device)
            for j in range(cfg.ports):
                detector_values_E = torch.stack(self.grid.detectors[j].detector_values()["E"], dim=0)
                detector_intensity = torch.mean(detector_values_E[:, :, -1].pow(2), dim=1)
                output = torch.cat((output, detector_intensity.unsqueeze(1)), dim=1)
            batch_output = torch.cat((batch_output, output.unsqueeze(0)), dim=0)
        return batch_output
