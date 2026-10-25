"""Basic dataset statistics for the Checkpoint 1 memo.

Writes results/checkpoint1/dataset_stats.json and dataset_stats.md with, per language/split:
sentence and word counts, entity counts before and after label mapping, XLM-R sub-word
fertility and truncation at max_length, and lexical overlap of Swahili entities with English train.

Usage:  python scripts/dataset_stats.py [--model xlm-roberta-base] [--max-length 256]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frozenweights.data import ENTITY_TYPES, REPO_ROOT, SPLITS, load_split  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "checkpoint1"


def entity_spans(tokens, tags):
    spans, cur = [], None
    for tok, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if cur:
                spans.append(cur)
            cur = (tag[2:], [tok])
        elif tag.startswith("I-") and cur and cur[0] == tag[2:]:
            cur[1].append(tok)
        else:
            if cur:
                spans.append(cur)
            cur = None
    if cur:
        spans.append(cur)
    return [(t, " ".join(w)) for t, w in spans]


def split_stats(rows, tokenizer, max_length):
    n_words = [len(r["tokens"]) for r in rows]
    orig = Counter(t[2:] for r in rows for t in r["orig_tags"] if t.startswith("B-"))
    mapped = Counter(t[2:] for r in rows for t in r["tags"] if t.startswith("B-"))
    enc = tokenizer([r["tokens"] for r in rows], is_split_into_words=True, add_special_tokens=True)
    n_sub = np.array([len(ids) for ids in enc["input_ids"]])
    n_sub_words = n_sub - 2  # strip <s> </s>
    return {
        "sentences": len(rows),
        "words": int(sum(n_words)),
        "words_per_sentence_mean": float(np.mean(n_words)),
        "words_per_sentence_max": int(max(n_words)),
        "entities_original": dict(sorted(orig.items())),
        "entities_mapped": {t: mapped.get(t, 0) for t in ENTITY_TYPES},
        "sentences_with_entity": int(sum(any(t != "O" for t in r["tags"]) for r in rows)),
        "subwords_per_word": float(n_sub_words.sum() / sum(n_words)),
        "subwords_per_sentence_p95": float(np.percentile(n_sub, 95)),
        "subwords_per_sentence_max": int(n_sub.max()),
        f"truncated_at_{max_length}": int((n_sub > max_length).sum()),
    }


def to_markdown(stats, overlap, max_length):
    lines = ["| Lang | Split | Sent. | Words | Words/sent | PER | ORG | LOC | Dropped (MISC/DATE) | Sub-words/word | p95 sub-words/sent | Truncated @%d |" % max_length,
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for lang, splits in stats.items():
        for split, s in splits.items():
            m = s["entities_mapped"]
            dropped = sum(v for k, v in s["entities_original"].items() if k not in ENTITY_TYPES)
            lines.append(f"| {lang} | {split} | {s['sentences']:,} | {s['words']:,} | {s['words_per_sentence_mean']:.1f} | "
                         f"{m['PER']:,} | {m['ORG']:,} | {m['LOC']:,} | {dropped:,} | {s['subwords_per_word']:.2f} | "
                         f"{s['subwords_per_sentence_p95']:.0f} | {s[f'truncated_at_{max_length}']} |")
    lines += ["", "Swahili test entities whose exact surface form appears as an entity in English train:", ""]
    for t, v in overlap.items():
        lines.append(f"- {t}: {v['seen']}/{v['total']} unique ({100 * v['seen'] / max(v['total'], 1):.1f}%)")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = {lang: {split: load_split(lang, split) for split in SPLITS} for lang in ("en", "sw")}
    stats = {lang: {split: split_stats(rows, tokenizer, args.max_length) for split, rows in splits.items()}
             for lang, splits in data.items()}

    en_ents = {(t, s) for r in data["en"]["train"] for t, s in entity_spans(r["tokens"], r["tags"])}
    sw_ents = {(t, s) for r in data["sw"]["test"] for t, s in entity_spans(r["tokens"], r["tags"])}
    overlap = {t: {"seen": sum(1 for e in sw_ents if e[0] == t and e in en_ents), "total": sum(1 for e in sw_ents if e[0] == t)}
               for t in ENTITY_TYPES}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dataset_stats.json").write_text(json.dumps({"tokenizer": args.model, "stats": stats, "sw_test_entity_overlap_with_en_train": overlap}, indent=2))
    md = to_markdown(stats, overlap, args.max_length)
    (OUT_DIR / "dataset_stats.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
