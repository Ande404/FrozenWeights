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
  metrics.py             seqeval entity-level P/R/F1 (strict, IOB2), per-type scores, paired bootstrap
  models.py              build_model(method) on a shared XLM-R + token-classification head
  cost.py                parameter counts, peak memory, timer, checkpoint size
tests/                   unit tests (pytest tests/)
outputs/                 model checkpoints (git-ignored; on Colab this is on Google Drive)
```

## Setup

**Colab (main compute):** open `notebooks/01_baselines.ipynb` in Colab, choose a GPU runtime, and run the cells
in order. The first cell mounts Drive, clones this repo, and installs `requirements.txt`. Checkpoints go to
`MyDrive/FrozenWeights/outputs/` and run results to `MyDrive/FrozenWeights/results/checkpoint1/`, so a dropped
session doesn't lose them. Nothing needs configuring: just allow the Drive pop-up. The notebook clones the
pushed `main` branch, so push your changes before running.

Direct link: <https://colab.research.google.com/github/Ande404/FrozenWeights/blob/main/notebooks/01_baselines.ipynb>

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

## Methods

Every method starts from the same pretrained **XLM-R base** encoder (~277M parameters) with the same linear
token-classification head, which maps each word's vector to one of 7 tags (O, B/I-PER, B/I-ORG, B/I-LOC).
The methods differ only in **which weights training may update**, and that difference is what we are measuring.

| `METHOD` | What is trained | Trainable params | Saved per task | Why it's in the study |
|---|---|---:|---:|---|
| `full_ft` | All XLM-R weights + head | 277.5M (100%) | ~1.1 GB | **Upper baseline.** Usually the best accuracy, but every task or language needs its own full model copy. |
| `head_only` | Only the head; XLM-R frozen | 5,383 (0.002%) | ~21 KB | **Lower baseline.** Shows how much NER information the frozen pretrained vectors already carry. |
| `lora` | Low-rank updates (r=16) on attention + feed-forward projections, + head; XLM-R frozen | 2.66M (0.95%) | ~10 MB | **Main PEFT method.** Does training ~1% of the parameters get close to `full_ft`? |
| `qlora` *(planned)* | Same as LoRA, but the frozen encoder is stored in 4-bit | ≈ LoRA | ≈ LoRA | Same idea with much less GPU memory. CUDA only (`bitsandbytes`). |
| `adapters` *(planned)* | Small bottleneck layers inserted in every transformer block, + head | ~1% | a few MB | The classic PEFT alternative to LoRA (Houlsby et al., 2019; AdapterHub). |
| `prompt_tuning` *(optional)* | A few learned vectors prepended to the input, + head | < 0.1% | KB | Extreme low-parameter end; often weak on token-level tasks. |

LoRA in one line: for a frozen weight matrix *W* it learns two thin matrices *B* (d × r) and *A* (r × d), so the
layer computes with *W + B·A*. Only *A* and *B* are trained and saved; *W* is never touched.

**Running a method.** Each method has a config in `configs/<METHOD>.yaml` (learning rate, epochs, batch size,
LoRA rank, and so on). In the notebook's config cell, set:

```python
METHOD = "lora"     # full_ft | head_only | lora
SEED = 1            # we report seeds 1, 2, 3
MAX_TRAIN = None    # e.g. 500 for a 2-minute smoke test
```

Then run the remaining cells in order. Each run writes:

- `results/checkpoint1/<METHOD>_seed<SEED>.json`: config, English/Swahili metrics, cost (params, peak memory,
  time, checkpoint size, GPU), and library versions
- `results/checkpoint1/<METHOD>_seed<SEED>_sw_test_predictions.jsonl`: Swahili gold vs. predicted tags for error analysis
- `outputs/<METHOD>_seed<SEED>/`: trainer checkpoints and `trainable.safetensors` (git-ignored)

The learning rate differs by method because frozen-encoder methods train far fewer weights and need larger
steps (full FT 2e-5, LoRA 5e-4, head-only 1e-3). The search grid for each method is listed in its config file.

Compare methods only on runs from the same GPU type, because time and memory are part of the result.

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
