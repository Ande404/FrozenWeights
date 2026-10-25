"""Unit tests for IOB handling, label mapping, sub-word alignment, metrics and few-shot sampling.

Run with:  pytest tests/   (the alignment test downloads the XLM-R tokenizer once)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frozenweights.data import (  # noqa: E402
    FEWSHOT_SIZES, LABEL2ID, map_to_shared_labels, read_conll, sample_fewshot_indices, to_iob2, tokenize_and_align,
)
from frozenweights.metrics import ner_scores, paired_bootstrap  # noqa: E402


def test_iob1_to_iob2():
    assert to_iob2(["I-PER", "I-PER", "O", "I-LOC"]) == ["B-PER", "I-PER", "O", "B-LOC"]
    # type change inside a run starts a new entity
    assert to_iob2(["B-ORG", "I-LOC"]) == ["B-ORG", "B-LOC"]


def test_iob2_is_unchanged():
    tags = ["B-PER", "I-PER", "O", "B-LOC", "B-LOC", "I-LOC"]
    assert to_iob2(tags) == tags


def test_label_mapping_drops_misc_and_date():
    assert map_to_shared_labels(["B-MISC", "I-MISC", "B-DATE", "B-PER", "I-ORG", "O"]) == ["O", "O", "O", "B-PER", "I-ORG", "O"]


def test_read_conll_skips_docstart(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("-DOCSTART- -X- -X- O\n\nEU NNP B-NP B-ORG\nrejects VBZ B-VP O\n\nPeter NNP B-NP B-PER\n")
    assert read_conll(f) == [
        {"tokens": ["EU", "rejects"], "tags": ["B-ORG", "O"]},
        {"tokens": ["Peter"], "tags": ["B-PER"]},
    ]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("xlm-roberta-base")


def test_label_on_first_subword_only(tokenizer):
    row = {"tokens": ["Wawakilishi", "John", "Lewis"], "tags": ["O", "B-PER", "I-PER"]}
    enc = tokenize_and_align([row], tokenizer)
    labels, word_ids = enc["labels"][0], enc.word_ids(0)
    # exactly one labelled position per word, and it carries that word's tag
    labelled = [(w, l) for w, l in zip(word_ids, labels) if l != -100]
    assert labelled == [(0, LABEL2ID["O"]), (1, LABEL2ID["B-PER"]), (2, LABEL2ID["I-PER"])]
    assert labels[0] == -100 and labels[-1] == -100  # <s>, </s>


def test_strict_span_matching():
    gold = [["B-PER", "I-PER", "O", "B-LOC"]]
    assert ner_scores(gold, gold)["f1"] == 1.0
    # partial span (boundary error) counts as both a false positive and a false negative
    s = ner_scores(gold, [["B-PER", "O", "O", "B-LOC"]])
    assert s["precision"] == 0.5 and s["recall"] == 0.5
    # right span, wrong type is also wrong
    assert ner_scores(gold, [["B-ORG", "I-ORG", "O", "B-LOC"]])["LOC_f1"] == 1.0


def test_paired_bootstrap_detects_better_system():
    gold = [["B-PER", "O", "B-LOC"]] * 50
    good, bad = gold, [["O", "O", "B-LOC"]] * 50
    r = paired_bootstrap(gold, good, bad, n_resamples=200)
    assert r["delta_f1"] > 0 and r["p_value"] == 0.0


def test_fewshot_samples_are_nested_and_deterministic():
    small, large = FEWSHOT_SIZES
    for seed in (1, 2, 3):
        a, b = sample_fewshot_indices(6593, small, seed), sample_fewshot_indices(6593, large, seed)
        assert len(a) == small and len(b) == large and set(a) <= set(b)
        assert a == sample_fewshot_indices(6593, small, seed)
