"""Tests for the Checkpoint 3 additions: few-shot training data, run naming, new methods, parameter counting.

Run with:  pytest tests/   (data tests need `python scripts/download_data.py` first; the adapters test
downloads XLM-R once; the QLoRA test only runs on a CUDA machine)
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frozenweights import cost, data  # noqa: E402

needs_data = pytest.mark.skipif(not (data.PROCESSED_DIR / "en" / "train.jsonl").exists(),
                                reason="run scripts/download_data.py first")


def load_runner():
    spec = importlib.util.spec_from_file_location("run_experiments", ROOT / "scripts" / "run_experiments.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_data
def test_training_rows_add_the_seeds_swahili_sample():
    en = data.load_split("en", "train")
    assert data.training_rows("zs", seed=2) == en
    for setting, n in (("fs100", 100), ("fs1000", 1000)):
        rows = data.training_rows(setting, seed=2)
        assert len(rows) == len(en) + n
        assert rows[: len(en)] == en and rows[len(en):] == data.load_fewshot(n, 2)


@needs_data
def test_fewshot_100_rows_are_inside_the_1000_rows():
    for seed in (1, 2, 3):
        small = {tuple(r["tokens"]) for r in data.load_fewshot(100, seed)}
        large = {tuple(r["tokens"]) for r in data.load_fewshot(1000, seed)}
        assert small <= large


def test_run_names_keep_checkpoint2_zero_shot_names():
    r = load_runner()
    assert r.final_name("lora", 1) == "lora_seed1"
    assert r.final_name("lora", 2, "fs100") == "lora_fs100_seed2"
    assert r.setting_glob("lora", "zs") == "lora_seed?.json"
    assert r.setting_glob("lora", "fs1000") == "lora_fs1000_seed?.json"
    # the zero-shot glob must not pick up few-shot files
    assert not Path("lora_fs100_seed1.json").match(r.setting_glob("lora", "zs"))


def test_count_parameters_unpacks_4bit_weights():
    packed = torch.nn.Parameter(torch.zeros(6, 1), requires_grad=False)  # 12 logical 4-bit values, 2 per byte
    packed.quant_state = SimpleNamespace(shape=torch.Size([3, 4]))
    head = torch.nn.Linear(2, 2)  # 6 trainable

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w, self.head = packed, head

    c = cost.count_parameters(M())
    assert c["total_params"] == 12 + 6 and c["trainable_params"] == 6


def test_adapters_train_only_adapters_and_classifier():
    pytest.importorskip("adapters")
    from frozenweights import models

    m = models.build_model("adapters", "xlm-roberta-base", {})
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert {"classifier.weight", "classifier.bias"} <= set(trainable)
    assert all(".adapters.ner." in n or n.startswith("classifier.") for n in trainable)
    assert cost.count_parameters(m)["trainable_params"] == 1_794_439  # 12 layers x 2 adapters (768->48->768) + head


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QLoRA needs CUDA (bitsandbytes)")
def test_qlora_trains_lora_and_classifier_on_a_4bit_encoder():
    from frozenweights import models

    m = models.build_model("qlora", "xlm-roberta-base", {})
    c = cost.count_parameters(m)
    assert c["trainable_params"] == 2_659_591  # same as LoRA
    assert 270e6 < c["total_params"] < 290e6  # logical size, not the packed 4-bit storage
