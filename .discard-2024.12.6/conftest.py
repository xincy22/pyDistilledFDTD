import os
import sys
from pathlib import Path

# 获取项目根目录
root_dir = str(Path(__file__).parent.absolute())

# 确保项目根目录在Python路径的最前面
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 打印路径以便调试
print("Project root added to Python path:", root_dir)
print("Updated Python path:", sys.path)