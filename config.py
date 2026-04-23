import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

BASE_DIR = "/mnt/share/viplab/zhous"
os.makedirs(BASE_DIR, exist_ok=True)