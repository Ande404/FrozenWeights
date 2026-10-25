"""Entity-level NER metrics with seqeval (strict span + type match, IOB2 scheme)."""

from __future__ import annotations

import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2

from .data import ENTITY_TYPES, ID2LABEL


def align_predictions(logits: np.ndarray, label_ids: np.ndarray) -> tuple[list[list[str]], list[list[str]]]:
    """Drop -100 positions and convert ids back to tag strings, one list per sentence."""
    pred_ids = logits.argmax(-1) if logits.ndim == 3 else logits
    y_true, y_pred = [], []
    for p_row, l_row in zip(pred_ids, label_ids):
        keep = l_row != -100
        y_true.append([ID2LABEL[int(i)] for i in l_row[keep]])
        y_pred.append([ID2LABEL[int(i)] for i in p_row[keep]])
    return y_true, y_pred


def ner_scores(y_true: list[list[str]], y_pred: list[list[str]]) -> dict:
    """Micro P/R/F1 plus per-type P/R/F1/support. Strict mode: a span counts only if exact."""
    kw = dict(mode="strict", scheme=IOB2, zero_division=0)
    out = {
        "precision": precision_score(y_true, y_pred, **kw),
        "recall": recall_score(y_true, y_pred, **kw),
        "f1": f1_score(y_true, y_pred, **kw),
    }
    report = classification_report(y_true, y_pred, output_dict=True, **kw)
    for t in ENTITY_TYPES:
        r = report.get(t, {})
        out[f"{t}_precision"] = r.get("precision", 0.0)
        out[f"{t}_recall"] = r.get("recall", 0.0)
        out[f"{t}_f1"] = r.get("f1-score", 0.0)
        out[f"{t}_support"] = int(r.get("support", 0))
    return out


def paired_bootstrap(y_true, pred_a, pred_b, n_resamples: int = 1000, seed: int = 0) -> dict:
    """Paired bootstrap over sentences for the F1 difference between systems A and B.

    Both systems are scored on the same resampled sentences each round. `p_value` is the
    one-sided share of rounds where A does not beat B (small = A is reliably better).
    """
    kw = dict(mode="strict", scheme=IOB2, zero_division=0)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_resamples)
    for k in range(n_resamples):
        idx = rng.integers(0, n, n)
        t = [y_true[i] for i in idx]
        deltas[k] = f1_score(t, [pred_a[i] for i in idx], **kw) - f1_score(t, [pred_b[i] for i in idx], **kw)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_f1": f1_score(y_true, pred_a, **kw) - f1_score(y_true, pred_b, **kw),
        "ci95": (float(lo), float(hi)),
        "p_value": float((deltas <= 0).mean()),
        "n_resamples": n_resamples,
    }


def compute_metrics(eval_pred) -> dict:
    """`compute_metrics` hook for transformers.Trainer."""
    logits, label_ids = eval_pred
    return ner_scores(*align_predictions(logits, label_ids))
