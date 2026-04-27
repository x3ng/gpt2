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

- **OpenAI WebText** — local JSONL files under `dataset/gpt-2-output-dataset/data/`
- **WikiText-103** — loaded from HuggingFace (`Salesforce/wikitext`, `wikitext-103-raw-v1`)

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
