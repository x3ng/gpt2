import os
import config  # load env before importing datasets/hf

from config import BASE_DIR
from datasets import load_dataset, concatenate_datasets


def _tokenize(dataset, tokenizer, text_col="text"):
    """Tokenize a dataset, removing the original text column."""
    def tokenize_fn(examples):
        return tokenizer(examples[text_col])
    return dataset.map(tokenize_fn, batched=True, remove_columns=[text_col])


def _chunk(dataset, block_size):
    """Concatenate-and-chunk a tokenized dataset into fixed-length blocks."""
    # Drop non-list columns before chunking
    cols_to_remove = [
        col for col in dataset.column_names
        if col not in ("input_ids", "attention_mask")
    ]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [concatenated[k][i : i + block_size] for i in range(0, total_len, block_size)]
            for k in concatenated.keys()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(group_texts, batched=True)
    dataset.set_format("torch")
    return dataset


def load_webtext(tokenizer, block_size=1024):
    """Load OpenAI WebText from local JSONL, tokenize & chunk."""
    data_dir = os.path.join(BASE_DIR, "dataset", "gpt-2-output-dataset", "data")

    train_ds = load_dataset("json", data_files=os.path.join(data_dir, "webtext.train.jsonl"), split="train")
    valid_ds = load_dataset("json", data_files=os.path.join(data_dir, "webtext.valid.jsonl"), split="train")

    train_ds = _tokenize(train_ds, tokenizer)
    valid_ds = _tokenize(valid_ds, tokenizer)

    train_ds = _chunk(train_ds, block_size)
    valid_ds = _chunk(valid_ds, block_size)

    return train_ds, valid_ds


def load_wikitext(tokenizer, block_size=1024):
    """Load WikiText-103-raw-v1 from HuggingFace, tokenize & chunk."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    train_ds = _tokenize(ds["train"], tokenizer)
    valid_ds = _tokenize(ds["validation"], tokenizer)

    train_ds = _chunk(train_ds, block_size)
    valid_ds = _chunk(valid_ds, block_size)

    return train_ds, valid_ds


def load_combined_datasets(tokenizer, block_size=1024):
    """Load WebText + WikiText, each chunked, then combine."""
    wt_train, wt_valid = load_webtext(tokenizer, block_size)
    wiki_train, wiki_valid = load_wikitext(tokenizer, block_size)

    train_ds = concatenate_datasets([wt_train, wiki_train])
    valid_ds = concatenate_datasets([wt_valid, wiki_valid])

    print(f"✅ 合并训练集: {len(train_ds)} 条 (block_size={block_size})")
    print(f"✅ 合并验证集: {len(valid_ds)} 条 (block_size={block_size})")

    return train_ds, valid_ds
