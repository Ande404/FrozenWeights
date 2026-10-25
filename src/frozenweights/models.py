"""Model construction. Every method shares the same XLM-R encoder and the same token-classification head."""

from __future__ import annotations

from transformers import AutoModelForTokenClassification

from .data import ID2LABEL, LABEL2ID

METHODS = ("full_ft", "head_only", "lora")  # qlora / adapters / prompt_tuning added next


def build_model(method: str, model_name: str = "xlm-roberta-base", cfg: dict | None = None):
    cfg = cfg or {}
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID
    )
    if method == "full_ft":
        pass  # everything trainable
    elif method == "head_only":
        # Freeze the encoder; only the dropout + linear classifier on top is trained.
        for p in model.base_model.parameters():
            p.requires_grad = False
    elif method == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        lora = LoraConfig(
            task_type=TaskType.TOKEN_CLS,  # keeps the classifier trainable (modules_to_save)
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.1),
            target_modules=cfg.get("lora_target_modules", ["query", "key", "value", "dense"]),
        )
        model = get_peft_model(model, lora)
    else:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
    return model
