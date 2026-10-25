"""Dataset reading, label mapping, few-shot sampling and tokenization.

Both corpora are stored as CoNLL-style text (one token per line, blank line between
sentences). `scripts/download_data.py` converts them to JSONL under data/processed/,
which is what the rest of the code reads.

Label space
-----------
CoNLL-2003 uses PER/ORG/LOC/MISC; MasakhaNER 2.0 uses PER/ORG/LOC/DATE. We train and
evaluate on the intersection, PER/ORG/LOC, mapping MISC and DATE to O. Otherwise
the model would be scored on a type (DATE) it was never shown in English, and it would
learn a type (MISC) that has no Swahili gold labels.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

ENTITY_TYPES = ["PER", "ORG", "LOC"]
LABELS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

LANGS = {"en": "CoNLL-2003 (English)", "sw": "MasakhaNER 2.0 (Swahili)"}
SPLITS = ("train", "dev", "test")


# --------------------------------------------------------------------------- reading

def read_conll(path: str | Path, tag_col: int = -1) -> list[dict]:
    """Parse a CoNLL file into [{"tokens": [...], "tags": [...]}]. Skips -DOCSTART- lines."""
    sentences, tokens, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                if tokens:
                    sentences.append({"tokens": tokens, "tags": tags})
                    tokens, tags = [], []
                continue
            cols = line.split()
            tokens.append(cols[0])
            tags.append(cols[tag_col])
    if tokens:
        sentences.append({"tokens": tokens, "tags": tags})
    return sentences


def to_iob2(tags: list[str]) -> list[str]:
    """Convert IOB1 (I- may start an entity) to IOB2 (every entity starts with B-). No-op on IOB2."""
    out = []
    for i, tag in enumerate(tags):
        if tag.startswith("I-"):
            prev = out[i - 1] if i else "O"
            if prev == "O" or prev[2:] != tag[2:]:
                tag = "B-" + tag[2:]
        out.append(tag)
    return out


def map_to_shared_labels(tags: list[str]) -> list[str]:
    """Map any tag whose type is outside ENTITY_TYPES (MISC, DATE, ...) to O."""
    return [t if t != "O" and t[2:] in ENTITY_TYPES else "O" for t in tags]


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_split(lang: str, split: str, processed_dir: str | Path = PROCESSED_DIR) -> list[dict]:
    """Load a processed split. Rows have tokens, tags (shared label space) and orig_tags."""
    return read_jsonl(Path(processed_dir) / lang / f"{split}.jsonl")


# --------------------------------------------------------------------------- few-shot

FEWSHOT_SIZES = (100, 1000)
FEWSHOT_SEEDS = (1, 2, 3)


def sample_fewshot_indices(n_total: int, size: int, seed: int) -> list[int]:
    return sorted(random.Random(seed).sample(range(n_total), size))


def load_fewshot(size: int, seed: int, processed_dir: str | Path = PROCESSED_DIR) -> list[dict]:
    """Return the fixed Swahili few-shot sample written by scripts/make_fewshot_splits.py."""
    idx = json.loads((SPLITS_DIR / f"sw_fewshot_{size}_seed{seed}.json").read_text())["indices"]
    train = load_split("sw", "train", processed_dir)
    return [train[i] for i in idx]


# --------------------------------------------------------------------------- tokenization

def tokenize_and_align(rows: list[dict], tokenizer, max_length: int = 256) -> dict:
    """Tokenize pre-split words and put each word's label on its first sub-word only.

    Remaining sub-words and special tokens get -100 so the loss and the metric ignore
    them. Predictions are then read back at first-sub-word positions (see metrics.py).
    """
    enc = tokenizer(
        [r["tokens"] for r in rows],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    all_labels = []
    for i, r in enumerate(rows):
        word_ids = enc.word_ids(batch_index=i)
        labels, prev = [], None
        for w in word_ids:
            if w is None or w == prev:
                labels.append(-100)
            else:
                labels.append(LABEL2ID[r["tags"][w]])
            prev = w
        all_labels.append(labels)
    enc["labels"] = all_labels
    return enc


def to_hf_dataset(rows: list[dict], tokenizer, max_length: int = 256):
    """Tokenized `datasets.Dataset` ready for a token-classification Trainer."""
    from datasets import Dataset

    enc = tokenize_and_align(rows, tokenizer, max_length)
    return Dataset.from_dict({k: enc[k] for k in ("input_ids", "attention_mask", "labels")})
