# FrozenWeights

Parameter-efficient fine-tuning (PEFT) for **zero-shot cross-lingual NER**: train XLM-R on English
(CoNLL-2003), evaluate on Swahili (MasakhaNER 2.0), and compare full fine-tuning, head-only, LoRA,
QLoRA, adapters and prompt tuning on both F1 and cost.

Andrew Luwaga · Brindha Sivakumar — UM-Dearborn NLP course project, Fall 2026.

## Research questions

1. **Performance:** in English→Swahili zero-shot transfer, how close does each PEFT method get to full fine-tuning?
2. **Cost:** trainable parameters, peak GPU memory, training time and checkpoint size; which method gives the best F1 per unit of cost?
3. **Target data:** does the ranking change with 100 or 1,000 labelled Swahili sentences mixed into training?

## Repository layout

```
configs/                 one YAML per method (hyperparameters + LR search grid)
data/
  raw/                   downloaded CoNLL files            (git-ignored)
  processed/{en,sw}/     train/dev/test.jsonl, shared labels (git-ignored)
  splits/                fixed Swahili few-shot sample indices (committed)
notebooks/
  01_baselines.ipynb     full fine-tuning / head-only, run cell by cell (Colab or local)
results/checkpoint1/     dataset stats, per-run metric + cost JSON, Swahili predictions
scripts/
  download_data.py       fetch + convert both datasets
  make_fewshot_splits.py write data/splits/*.json
  dataset_stats.py       dataset statistics table
src/frozenweights/
  data.py                CoNLL reader, label mapping, few-shot loading, sub-word alignment
  metrics.py             seqeval entity-level P/R/F1 (strict, IOB2), per-type scores
  models.py              build_model(method) on a shared XLM-R + token-classification head
  cost.py                parameter counts, peak memory, timer, checkpoint size
outputs/                 model checkpoints (git-ignored; on Colab this is on Google Drive)
```

## Setup

**Colab (main compute):** open `notebooks/01_baselines.ipynb` in Colab, choose a GPU runtime, and run the cells
in order. The first cell mounts Drive, clones this repo, and installs `requirements.txt`. Checkpoints go to
`MyDrive/FrozenWeights/outputs/` so a dropped session doesn't lose them. The repo has to be pushed and readable
from Colab (public, or clone with a token).

**Local:**

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py      # ~6 MB, writes data/raw + data/processed
python scripts/make_fewshot_splits.py
python scripts/dataset_stats.py
jupyter lab notebooks/01_baselines.ipynb
```

Apple Silicon runs on MPS. That's fine for debugging, but reported numbers should come from Colab so GPU type
and memory are comparable.

## Data

| | Source | Train | Dev | Test |
|---|---|---:|---:|---:|
| English | CoNLL-2003 ([eriktks/conll2003](https://huggingface.co/datasets/eriktks/conll2003)) | 14,041 | 3,250 | 3,453 |
| Swahili | MasakhaNER 2.0 `swa` ([masakhane/masakhaner2](https://huggingface.co/datasets/masakhane/masakhaner2)) | 6,593 | 942 | 1,883 |

Both Hub repos are loading scripts, which recent `datasets` releases no longer execute.
`download_data.py` downloads the same raw files those scripts use (deepai.org mirror of CoNLL-2003; the
MasakhaNER GitHub repo) and checks the sentence counts.

**Shared label space.** CoNLL has PER/ORG/LOC/MISC and MasakhaNER has PER/ORG/LOC/DATE. We train and evaluate on
**PER, ORG, LOC** and map MISC and DATE to `O`. The original tags are kept in each row as `orig_tags`.

## Protocol

- Model selection (epochs, learning rate) on **English dev** only; no Swahili labels beyond the few-shot budget.
- Swahili test is scored once per run. Entity-level micro F1 (seqeval, strict) is the primary metric, alongside P/R and per-type F1.
- 3 seeds per configuration, reported as mean ± std; paired bootstrap for key comparisons.
- Per-run cost: trainable params, peak memory, wall-clock training time, size of the saved trainable weights.
