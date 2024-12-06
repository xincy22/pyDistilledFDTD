from .base_config import DATA_DIR
import os
# 数据配置
PCA_COMPONENTS = 10
BATCH_SIZE = 16
CORE_SET_ETA = 0.1
CORE_SET_METHOD = 'greedy'

# 数据目录
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
CACHE_DATA_DIR = os.path.join(DATA_DIR, 'cache')

# 创建目录
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DATA_DIR, exist_ok=True)
