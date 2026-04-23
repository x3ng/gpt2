# %% ===================== imports && env =====================
import os
import config # load huggingface config, BASE_DIR

from config import BASE_DIR

# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# 
# BASE_DIR = "/mnt/share/viplab/zhous"
# os.makedirs(BASE_DIR, exist_ok=True)

import torch
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# %% ===================== model config =====================
config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,        # 隐变量长度 768
    n_layer=12,        # 12 层
    n_head=12,         # 12 头
    activation_function="gelu",
)

# 从头随机初始化
model = GPT2LMHeadModel(config)
print(f"✅ 模型总参数量: {model.num_parameters() / 1e6:.2f}M")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# %% ===================== load data =====================
def load_and_tokenize(file_path, tokenizer, block_size=128):
    # 1. 加载原始jsonl数据
    dataset = load_dataset("json", data_files=file_path, split="train")

    # 2. 分词：文本 -> token ids
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=block_size,
            padding="max_length",
        )
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # 3. 添加标签
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    tokenized = tokenized.map(add_labels, batched=True)

    tokenized.set_format("torch")

    return tokenized

# 加载 OpenAI WebText 数据
train_dataset = load_and_tokenize(f"{BASE_DIR}/gpt-2-output-dataset/data/webtext.train.jsonl", tokenizer)
valid_dataset = load_and_tokenize(f"{BASE_DIR}/gpt-2-output-dataset/data/webtext.valid.jsonl", tokenizer)

print(f"✅ 训练集数量: {len(train_dataset)}")
print(f"✅ 验证集数量: {len(valid_dataset)}")

# %% ===================== datt collator =====================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# %% ===================== train args =====================
training_args = TrainingArguments(
    output_dir=f"{BASE_DIR}/gpt2-117m-scratch",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    learning_rate=5e-4,
    warmup_steps=1000,
    fp16=True,

    # 验证配置
    eval_strategy="steps",
    eval_steps=50,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,

    logging_dir=f"{BASE_DIR}/gpt2-117m-scratch/logs",  # 日志保存路径
    report_to="tensorboard",                # 开启tensorboard
)

# %% ===================== config trainer =====================
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# %% ===================== train =====================
print("🚀 开始训练...")
trainer.train()
trainer.save_model(f"{BASE_DIR}/gpt2-117m-final")
print("✅ 训练完成，模型已保存")

# %% ===================== 9. 验证集评估 =====================
print("\n===== 最终验证集评估 =====")
eval_results = trainer.evaluate()
print(eval_results)

# 报告格式输出
loss = eval_results['eval_loss']
perp = torch.exp(torch.tensor(loss)).item()
print(f"\n📊 测试报告指标：")
print(f"验证损失 (loss): {loss:.4f}")
print(f"困惑度 (perplexity): {perp:.4f}")