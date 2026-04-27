# %% ===================== imports =====================
import os
import config

from config import BASE_DIR

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# %% ===================== VAR ====================
MODEL_PATH = os.path.join(BASE_DIR, "result", "gpt2-117m-final")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ 模型路径不存在: {MODEL_PATH}\n请先运行训练脚本生成模型。")

print(f"📂 正在从 {MODEL_PATH} 加载模型...")

# %% ===================== load model =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

print("✅ 模型加载完成")

# %% ===================== text generate =====================
print("\n===== 文本生成测试 =====")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

default_prompt = "Hello, world!"

user_input = input(f"请输入提示词 (enter使用默认: '{default_prompt}'): ")

prompt = user_input if user_input.strip() else default_prompt

print(f"📝 确认提示词: {prompt}")

output = generator(
    prompt,
    max_new_tokens=100,
    truncation=True,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
)

print(f"\n✨ 生成结果:\n{output[0]['generated_text']}")
