import torch

WAVELENGTH = 1550e-9 # 波长(m)
SPEED_LIGHT: float = 299_792_458.0 # 光速(m/s)
backend = "torch.cuda.float64" if torch.cuda.is_available() else "torch.float64"

### 仿真区域设置
# 整个区域
Nx = 400
Ny = 400
dx = 25e-9
simulation_step = 1000

# 输入输出端口
ports = 10
port_width = 20
ports_slice = [
    slice(82, 102), slice(106, 126), slice(130, 150), slice(154, 174), slice(178, 198),
    slice(202, 222), slice(226, 246), slice(250, 270), slice(274, 294), slice(298, 318)
]
source_loc = 75
detector_loc = 325

# 中心方形区域
center_size = 250
center_slice = slice(75, 325)
circle_count = 10
spacing = 25
border_spacing = 12
x_centers, y_centers = torch.meshgrid(
    torch.arange(border_spacing, center_size, spacing),
    torch.arange(border_spacing, center_size, spacing),
    indexing='ij'
)
x_centers, y_centers = x_centers.flatten(), y_centers.flatten()
