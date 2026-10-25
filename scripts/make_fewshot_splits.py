"""Write the fixed Swahili few-shot samples used in Experiment 2.

For each size in {100, 1000} and seed in {1, 2, 3}, sample sentence indices from the
MasakhaNER 2.0 Swahili *train* split (never dev/test). The index files are committed so
every method sees exactly the same sentences.

    data/splits/sw_fewshot_{size}_seed{seed}.json

Usage:  python scripts/make_fewshot_splits.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frozenweights.data import FEWSHOT_SEEDS, FEWSHOT_SIZES, SPLITS_DIR, load_split, sample_fewshot_indices  # noqa: E402

if __name__ == "__main__":
    train = load_split("sw", "train")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for size in FEWSHOT_SIZES:
        for seed in FEWSHOT_SEEDS:
            idx = sample_fewshot_indices(len(train), size, seed)
            ents = Counter(t[2:] for i in idx for t in train[i]["tags"] if t.startswith("B-"))
            out = SPLITS_DIR / f"sw_fewshot_{size}_seed{seed}.json"
            out.write_text(json.dumps({"size": size, "seed": seed, "source": "sw/train", "entities": dict(ents), "indices": idx}))
            print(f"{out.name}: {dict(sorted(ents.items()))}")
