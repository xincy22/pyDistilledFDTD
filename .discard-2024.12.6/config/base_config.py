from pathlib import Path
import os

# 获取项目根目录
ROOT_DIR = Path(os.path.dirname(os.path.dirname(__file__)))

# 基础路径配置
CACHE_DIR = os.path.join(ROOT_DIR, 'cache')

# 细分目录
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, 'model')
DATA_DIR = os.path.join(CACHE_DIR, 'data')
LOG_DIR = os.path.join(CACHE_DIR, 'logs')

# 创建目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)