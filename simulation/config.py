import torch

# 物理常数
WAVELENGTH = 1550e-9  # 波长(m)
SPEED_LIGHT: float = 299_792_458.0  # 光速(m/s)

# 设备配置
backend = "torch.cuda.float64" if torch.cuda.is_available() else "torch.float64"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 仿真区域设置
Nx = 400  # x方向网格数
Ny = 400  # y方向网格数
dx = 25e-9  # 网格间距
simulation_step = 1000  # 仿真步数

# 输入输出端口配置
ports = 10  # 端口数量
port_width = 20  # 端口宽度
ports_slice = [
    slice(82, 102), slice(106, 126), slice(130, 150),
    slice(154, 174), slice(178, 198), slice(202, 222),
    slice(226, 246), slice(250, 270), slice(274, 294),
    slice(298, 318)
]  # 端口位置
source_loc = 75  # 源位置
detector_loc = 325  # 探测器位置

# 中心方形区域配置
center_size = 250  # 中心区域大小
center_slice = slice(75, 325)  # 中心区域切片
circle_count = 10  # 圆形数量
spacing = 25  # 圆形间距
border_spacing = 12  # 边界间距

# 计算圆形中心位置
x_centers, y_centers = torch.meshgrid(
    torch.arange(border_spacing, center_size, spacing),
    torch.arange(border_spacing, center_size, spacing),
    indexing='ij'
)
x_centers = x_centers.flatten()
y_centers = y_centers.flatten() 