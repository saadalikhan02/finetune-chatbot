"""Configuration loading for training and LoRA settings.

Both configs are plain YAML so values can be tweaked without touching code.
These dataclasses just give the rest of the codebase typed, validated access
to those values instead of passing raw dicts around.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping")
    return data


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Construct a dataclass from a dict, rejecting unknown keys with a
    clear error rather than silently ignoring typos in a YAML file."""
    valid_keys = {f.name for f in fields(cls)}
    unknown = set(data) - valid_keys
    if unknown:
        raise ValueError(
            f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid_keys)}"
        )
    return cls(**data)


@dataclass
class LoraSettings:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingSettings:
    model_name: str = "google/gemma-3-1b-it"

    train_file: str = "data/train/train.jsonl"
    validation_file: str = "data/validation/validation.jsonl"

    output_dir: str = "outputs/gemma3-1b-corporate-lora"

    seed: int = 42

    num_train_epochs: float = 3
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    optim: str = "paged_adamw_8bit"

    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True

    max_seq_length: int = 1024
    packing: bool = False

    bf16: bool = True
    fp16: bool = False

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "auto"

    logging_steps: int = 5
    save_steps: int = 50
    eval_steps: int = 50
    save_total_limit: int = 2
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    report_to: str = "none"


def load_lora_config(path: str | Path) -> LoraSettings:
    return _build_dataclass(LoraSettings, load_yaml(path))


def load_training_config(path: str | Path) -> TrainingSettings:
    return _build_dataclass(TrainingSettings, load_yaml(path))


def print_training_config(training: TrainingSettings, lora: LoraSettings) -> None:
    print("## Training configuration")
    print(f"Model:                {training.model_name}")
    print(
        f"Quantization:         "
        f"{'4-bit QLoRA (' + training.bnb_4bit_quant_type + ')' if training.load_in_4bit else 'disabled'}"
    )
    print(f"LoRA rank:            {lora.r} (alpha={lora.lora_alpha}, dropout={lora.lora_dropout})")
    print(f"LoRA target modules:  {', '.join(lora.target_modules)}")
    print(f"Learning rate:        {training.learning_rate}")
    print(f"Epochs:               {training.num_train_epochs}")
    print(f"Batch size:           {training.per_device_train_batch_size} (eval: {training.per_device_eval_batch_size})")
    print(f"Gradient accumulation:{training.gradient_accumulation_steps}")
    print(f"Effective batch size: {training.per_device_train_batch_size * training.gradient_accumulation_steps}")
    print(f"Max sequence length:  {training.max_seq_length}")
    print(f"Output dir:           {training.output_dir}")
