""" Python 3D wave Simulator """

# from .grid import Grid
# from .sources import PointSource, LineSource, PlaneSource
# from .detectors import LineDetector, BlockDetector
# from .objects import Object, AbsorbingObject, AnisotropicObject
# from .boundaries import PeriodicBoundary, PML
# from .backend import backend
# from .backend import set_backend,set_device
# from .visualization import dB_map_2D, plot_detection
import os
import sys

sys.path.append(r"D:\研究生工作\Photonic-Computing\code\OpTorch_9_13\Sim")
# print(sys.path)
# # 获取当前文件的绝对路径
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # 将当前目录添加到Python的搜索路径中
# sys.path.append(current_dir)
import backend
from .init import *
from .backend import backend