"""Model construction. Every method shares the same XLM-R encoder and the same token-classification head."""

from __future__ import annotations

from transformers import AutoModelForTokenClassification

from .data import ID2LABEL, LABEL2ID

METHODS = ("full_ft", "head_only")  # lora / qlora / adapters / prompt_tuning added next


def build_model(method: str, model_name: str = "xlm-roberta-base"):
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID
    )
    if method == "full_ft":
        pass  # everything trainable
    elif method == "head_only":
        # Freeze the encoder; only the dropout + linear classifier on top is trained.
        for p in model.base_model.parameters():
            p.requires_grad = False
    else:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
    return model
