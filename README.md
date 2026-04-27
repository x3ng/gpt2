# GPT-2 117M Training

From-scratch training of a 117M-parameter GPT-2 language model using HuggingFace Transformers.

## Model

| Parameter | Value |
|-----------|-------|
| Params | ~117M |
| Hidden dim | 768 |
| Layers | 12 |
| Heads | 12 |
| Context length | 1024 |
| Vocab size | 50,257 |
| Activation | GELU |

## Datasets

- **OpenAI WebText** — from [gpt-2-output-dataset](https://github.com/openai/gpt-2-output-dataset), download `webtext.train.jsonl` and `webtext.valid.jsonl` into `dataset/gpt-2-output-dataset/data/`
- **WikiText-103** — from [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) (`wikitext-103-raw-v1`), auto-downloaded via HuggingFace Datasets API

## Usage

```bash
pip install torch transformers datasets numpy
```

```bash
# Train
python train.py

# Inference
python inference.py
```

## Project Structure

```
gpt2/
├── config.py       # Environment config (HF mirror, CUDA, paths)
├── dataset.py      # Data loading & preprocessing (WebText + WikiText)
├── train.py        # Training script
├── inference.py    # Text generation script
├── dataset/        # Training data (gitignored)
└── result/         # Checkpoints & final model (gitignored)
```

## Results

After 5 epochs on WebText + WikiText-103 with block_size=1024:

| Metric | Value |
|--------|-------|
| Eval loss | 3.47 |
| Perplexity | 32.2 |
| Training time | ~7.8h on RTX 4090 |

## Environment

- Python 3.12, PyTorch, HuggingFace Transformers
- Single NVIDIA RTX 4090 (40GB)
- HF mirror (`hf-mirror.com`) for China network

## License

MIT
