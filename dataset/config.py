import os

# 获取当前文件所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 原始数据目录
RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw_data')

# 处理后的数据目录
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')

# PCA数据文件
PCA_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'pca_data.pkl')

# 核心集数据文件
CORE_SET_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'core_set_data.pkl')
