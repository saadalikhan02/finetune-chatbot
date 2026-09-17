"""Trainer construction for supervised fine-tuning with TRL's SFTTrainer.

Isolated here so scripts/train.py stays a thin CLI wrapper, and so the
TRL/transformers version-specific argument names only need updating in one
place if a future TRL release changes them.
"""

from __future__ import annotations

from typing import Any

from .config import TrainingSettings
from .formatting import format_example_for_training


def format_dataset_for_sft(dataset: Any, tokenizer: Any) -> Any:
    """Map a ``{"messages": [...]}}`` dataset to a ``{"text": ...}`` dataset
    using the tokenizer's own chat template (see formatting.py)."""
    return dataset.map(
        lambda example: format_example_for_training(tokenizer, example),
        remove_columns=dataset.column_names,
    )


def build_sft_config(training_cfg: TrainingSettings, cuda_available: bool):
    """Build a trl.SFTConfig from our TrainingSettings.

    bf16/fp16 are forced off when no CUDA GPU is present, since mixed
    precision training targets GPU execution.
    """
    from trl import SFTConfig

    bf16 = training_cfg.bf16 and cuda_available
    fp16 = training_cfg.fp16 and cuda_available and not bf16

    return SFTConfig(
        output_dir=training_cfg.output_dir,
        seed=training_cfg.seed,
        num_train_epochs=training_cfg.num_train_epochs,
        learning_rate=training_cfg.learning_rate,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        warmup_ratio=training_cfg.warmup_ratio,
        weight_decay=training_cfg.weight_decay,
        max_grad_norm=training_cfg.max_grad_norm,
        optim=training_cfg.optim if cuda_available else "adamw_torch",
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=training_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        gradient_checkpointing=training_cfg.gradient_checkpointing,
        max_seq_length=training_cfg.max_seq_length,
        packing=training_cfg.packing,
        dataset_text_field="text",
        bf16=bf16,
        fp16=fp16,
        logging_steps=training_cfg.logging_steps,
        save_steps=training_cfg.save_steps,
        eval_steps=training_cfg.eval_steps,
        save_total_limit=training_cfg.save_total_limit,
        eval_strategy=training_cfg.eval_strategy,
        save_strategy=training_cfg.save_strategy,
        load_best_model_at_end=training_cfg.load_best_model_at_end,
        metric_for_best_model=training_cfg.metric_for_best_model,
        report_to=training_cfg.report_to,
    )


def build_trainer(
    model: Any,
    tokenizer: Any,
    sft_config: Any,
    train_dataset: Any,
    eval_dataset: Any,
):
    from trl import SFTTrainer

    return SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
