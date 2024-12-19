import os

# 获取当前文件所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 蒸馏数据目录
DISTILL_DATA_DIR = os.path.join(BASE_DIR, 'data')
