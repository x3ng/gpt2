# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Minimal GPT-2 (117M parameter) training and inference implementation using Hugging Face Transformers. Trains from scratch on OpenAI WebText + WikiText-103 datasets.

## Environment

- Python 3.12 (py312 conda/virtual environment)
- Shared GPU server — `CUDA_VISIBLE_DEVICES="3"` in config.py (only use 1 GPU)
- HuggingFace mirror: `https://hf-mirror.com` (China network)

## Key Files

- **config.py** — Environment config (CUDA device, HF mirror, BASE_DIR as project root)
- **dataset.py** — Data loading: WebText (local JSONL) + WikiText-103 (HuggingFace), concatenation-and-chunking
- **train.py** — Training pipeline: model config, data loading, Trainer setup
- **inference.py** — Interactive text generation with trained models

## Directory Structure

- `dataset/` — Training data (gpt-2-output-dataset JSONL files)
- `result/` — All training outputs (checkpoints, logs, final model)
  - `result/gpt2-117m-scratch/` — Trainer checkpoints + TensorBoard logs
  - `result/gpt2-117m-final/` — Final saved model

## Commands

```bash
# Install dependencies
pip install torch transformers datasets numpy

# Train GPT-2 from scratch
python train.py

# Run inference
python inference.py
```

## Architecture

- GPT-2 small: 12 layers, 12 heads, 768 hidden dim, 1024 context, GELU (~117M params)
- Block size: 512 tokens (concatenate-and-chunk, no padding waste)
- Effective batch size: 32 (16 per device × 2 gradient accumulation)
- AdamW (lr=5e-4), linear warmup 1000 steps, 5 epochs
- Eval every 500 steps, save every 1000 steps
