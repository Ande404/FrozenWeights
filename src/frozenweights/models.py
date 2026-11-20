"""Model construction. Every method shares the same XLM-R encoder and the same token-classification head."""

from __future__ import annotations

from transformers import AutoModelForTokenClassification

from .data import ID2LABEL, LABEL2ID

METHODS = ("full_ft", "head_only", "lora", "qlora", "adapters")  # prompt_tuning is optional / not implemented


def _lora_config(cfg: dict):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.TOKEN_CLS,  # keeps the classifier trainable (modules_to_save)
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.1),
        target_modules=cfg.get("lora_target_modules", ["query", "key", "value", "dense"]),
    )


def build_model(method: str, model_name: str = "xlm-roberta-base", cfg: dict | None = None):
    cfg = cfg or {}
    labels = dict(num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID)

    if method == "qlora":
        # Same LoRA as below, on an encoder whose linear layers are stored in 4-bit NF4 (CUDA only).
        # The classifier stays in full precision; embeddings are not linear layers, so bitsandbytes
        # leaves them unquantized (they are ~190M of XLM-R base's 278M parameters).
        import torch
        from peft import get_peft_model, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("qlora needs a CUDA GPU (bitsandbytes 4-bit kernels)")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["classifier"],
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, quantization_config=bnb, device_map={"": 0}, **labels
        )
        # No gradient checkpointing, so the only difference from `lora` is the quantized encoder.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        return get_peft_model(model, _lora_config(cfg))

    model = AutoModelForTokenClassification.from_pretrained(model_name, **labels)
    if method == "full_ft":
        pass  # everything trainable
    elif method == "head_only":
        # Freeze the encoder; only the dropout + linear classifier on top is trained.
        for p in model.base_model.parameters():
            p.requires_grad = False
    elif method == "lora":
        from peft import get_peft_model

        model = get_peft_model(model, _lora_config(cfg))
    elif method == "adapters":
        # Houlsby-style bottleneck adapters (AdapterHub): one after the attention block and one
        # after the feed-forward block in every layer. train_adapter() freezes everything else,
        # so the shared classifier is unfrozen explicitly.
        import adapters
        from adapters import DoubleSeqBnConfig

        adapters.init(model)
        model.add_adapter("ner", config=DoubleSeqBnConfig(reduction_factor=cfg.get("adapter_reduction_factor", 16)))
        model.train_adapter("ner")
        for p in model.classifier.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
    return model
