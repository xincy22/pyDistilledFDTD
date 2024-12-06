import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
CACHE_DATA_DIR = os.path.join(DATA_DIR, "cache")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# 日志目录
LOG_DIR = os.path.join(BASE_DIR, "logs")

for dir_path in [DATA_DIR, RAW_DATA_DIR, CACHE_DATA_DIR, PROCESSED_DATA_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)