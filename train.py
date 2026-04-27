# %% ===================== imports && env =====================
import os
import config  # load huggingface config, BASE_DIR

from config import BASE_DIR

import torch
from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from dataset import load_combined_datasets

# %% ===================== model config =====================
BLOCK_SIZE = 1024

config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    activation_function="gelu",
)

model = GPT2LMHeadModel(config)
print(f"✅ 模型总参数量: {model.num_parameters() / 1e6:.2f}M")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# %% ===================== load data =====================
train_dataset, valid_dataset = load_combined_datasets(tokenizer, block_size=BLOCK_SIZE)

# %% ===================== data collator =====================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# %% ===================== paths =====================
RESULT_DIR = os.path.join(BASE_DIR, "result", "gpt2-117m-scratch")
os.makedirs(RESULT_DIR, exist_ok=True)

# %% ===================== train args =====================
training_args = TrainingArguments(
    output_dir=RESULT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,  # effective batch = 16*2 = 32
    learning_rate=5e-4,
    warmup_steps=1000,
    fp16=True,

    eval_strategy="steps",
    eval_steps=500,   # 每500步评测一次，减少评测开销
    logging_steps=50,
    save_steps=1000,
    save_total_limit=3,
    load_best_model_at_end=True,

    logging_dir=os.path.join(RESULT_DIR, "logs"),
    report_to="tensorboard",
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

FINAL_MODEL_DIR = os.path.join(BASE_DIR, "result", "gpt2-117m-final")
trainer.save_model(FINAL_MODEL_DIR)
print(f"✅ 训练完成，模型已保存至 {FINAL_MODEL_DIR}")

# %% ===================== eval =====================
print("\n===== 最终验证集评估 =====")
eval_results = trainer.evaluate()
print(eval_results)

loss = eval_results["eval_loss"]
perp = torch.exp(torch.tensor(loss)).item()
print(f"\n📊 测试报告指标：")
print(f"验证损失 (loss): {loss:.4f}")
print(f"困惑度 (perplexity): {perp:.4f}")
