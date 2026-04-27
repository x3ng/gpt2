import os

# HuggingFace 镜像加速（国内网络）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 共享GPU，仅使用一块
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 项目根目录 = 当前仓库根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
