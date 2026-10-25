"""Download CoNLL-2003 (English) and MasakhaNER 2.0 Swahili and write processed JSONL.

The Hugging Face repos cited in the proposal (eriktks/conll2003, masakhane/masakhaner2)
are loading scripts, which `datasets>=4` no longer runs. We fetch the same raw files those
scripts point to and parse them ourselves:

    data/raw/conll2003/{train,valid,test}.txt
    data/raw/masakhaner2_swa/{train,dev,test}.txt
    data/processed/{en,sw}/{train,dev,test}.jsonl   # tokens, tags (PER/ORG/LOC), orig_tags

Usage:  python scripts/download_data.py
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frozenweights.data import DATA_DIR, PROCESSED_DIR, map_to_shared_labels, read_conll, to_iob2, write_jsonl  # noqa: E402

RAW_DIR = DATA_DIR / "raw"
CONLL_URL = "https://data.deepai.org/conll2003.zip"  # same source as eriktks/conll2003
MASAKHA_URL = "https://github.com/masakhane-io/masakhane-ner/raw/main/MasakhaNER2.0/data/swa/{}.txt"

# split name in our layout -> file name in the source
SOURCES = {
    "en": {"dir": RAW_DIR / "conll2003", "files": {"train": "train.txt", "dev": "valid.txt", "test": "test.txt"}},
    "sw": {"dir": RAW_DIR / "masakhaner2_swa", "files": {"train": "train.txt", "dev": "dev.txt", "test": "test.txt"}},
}
# Sentence counts from the dataset papers / cards; a mismatch means the download changed.
EXPECTED = {"en": {"train": 14041, "dev": 3250, "test": 3453}, "sw": {"train": 6593, "dev": 942, "test": 1883}}


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  exists  {dest.relative_to(DATA_DIR.parent)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetch   {url}")
    urllib.request.urlretrieve(url, dest)


def download() -> None:
    en_dir = SOURCES["en"]["dir"]
    zip_path = en_dir / "conll2003.zip"
    fetch(CONLL_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for name in SOURCES["en"]["files"].values():
            if not (en_dir / name).exists():
                z.extract(name, en_dir)

    for split, name in SOURCES["sw"]["files"].items():
        fetch(MASAKHA_URL.format(split), SOURCES["sw"]["dir"] / name)


def process() -> None:
    for lang, src in SOURCES.items():
        for split, name in src["files"].items():
            sents = read_conll(src["dir"] / name)
            rows = []
            for s in sents:
                orig = to_iob2(s["tags"])
                rows.append({"tokens": s["tokens"], "tags": map_to_shared_labels(orig), "orig_tags": orig})
            out = PROCESSED_DIR / lang / f"{split}.jsonl"
            write_jsonl(rows, out)
            exp = EXPECTED[lang][split]
            flag = "ok" if len(rows) == exp else f"MISMATCH (expected {exp})"
            print(f"  {lang}/{split:5s} {len(rows):6d} sentences  {flag}")


if __name__ == "__main__":
    print("Downloading raw files ...")
    download()
    print("Writing processed JSONL ...")
    process()
