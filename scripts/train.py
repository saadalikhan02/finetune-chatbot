#!/usr/bin/env python3
"""Fine-tune google/gemma-3-1b-it with QLoRA on the corporate chatbot dataset.

Loads config, validates the dataset, loads a 4-bit quantized base model,
attaches LoRA adapters, trains with TRL's SFTTrainer, and saves the LoRA
adapter (not a merged model) to the configured output directory.

Usage:
    python scripts/train.py --config configs/training.yaml --lora-config configs/lora.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corporate_chatbot.config import (  # noqa: E402
    load_lora_config,
    load_training_config,
    print_training_config,
)
from corporate_chatbot.dataset import load_and_validate, to_hf_dataset  # noqa: E402
from corporate_chatbot.model import (  # noqa: E402
    build_bnb_config,
    build_lora_config,
    load_base_model,
    load_tokenizer,
    prepare_model_for_qlora_training,
)
from corporate_chatbot.training import build_sft_config, build_trainer, format_dataset_for_sft  # noqa: E402
from corporate_chatbot.utils import (  # noqa: E402
    eprint,
    get_hf_token,
    load_dotenv_if_present,
    print_environment_report,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml", type=Path, help="Training config YAML")
    parser.add_argument("--lora-config", default="configs/lora.yaml", type=Path, help="LoRA config YAML")
    return parser.parse_args()


def main() -> int:
    load_dotenv_if_present()
    args = parse_args()

    hw = print_environment_report()

    training_cfg = load_training_config(args.config)
    lora_cfg = load_lora_config(args.lora_config)
    print()
    print_training_config(training_cfg, lora_cfg)

    if not hw.cuda_available:
        eprint(
            "\nERROR: no CUDA GPU detected. QLoRA training requires a CUDA GPU "
            "for 4-bit quantization via bitsandbytes. Run this on a machine "
            "with an NVIDIA GPU (e.g. Google Colab) instead of CPU-only hardware."
        )
        return 1

    set_seed(training_cfg.seed)

    print(f"\nValidating training set: {training_cfg.train_file}")
    train_result = load_and_validate(training_cfg.train_file)
    if not train_result.is_valid:
        eprint(f"ERROR: training set has {len(train_result.issues)} validation error(s):")
        for issue in train_result.issues:
            eprint(f"  - {issue}")
        return 1
    print(f"  {len(train_result.valid_records)} valid example(s)")

    print(f"Validating validation set: {training_cfg.validation_file}")
    val_result = load_and_validate(training_cfg.validation_file)
    if not val_result.is_valid:
        eprint(f"ERROR: validation set has {len(val_result.issues)} validation error(s):")
        for issue in val_result.issues:
            eprint(f"  - {issue}")
        return 1
    print(f"  {len(val_result.valid_records)} valid example(s)")

    hf_token = get_hf_token()

    print(f"\nLoading tokenizer: {training_cfg.model_name}")
    tokenizer = load_tokenizer(training_cfg.model_name, hf_token=hf_token)

    print(f"Loading base model: {training_cfg.model_name}")
    bnb_config = build_bnb_config(training_cfg, cuda_available=hw.cuda_available)
    model = load_base_model(training_cfg.model_name, bnb_config=bnb_config, hf_token=hf_token)

    print("Preparing model for k-bit training...")
    model = prepare_model_for_qlora_training(
        model, gradient_checkpointing=training_cfg.gradient_checkpointing
    )

    print("Attaching LoRA adapters...")
    from peft import get_peft_model

    lora_config = build_lora_config(lora_cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\nBuilding datasets...")
    train_dataset = to_hf_dataset(train_result.valid_records)
    eval_dataset = to_hf_dataset(val_result.valid_records)
    train_dataset = format_dataset_for_sft(train_dataset, tokenizer)
    eval_dataset = format_dataset_for_sft(eval_dataset, tokenizer)

    sft_config = build_sft_config(training_cfg, cuda_available=hw.cuda_available)
    trainer = build_trainer(model, tokenizer, sft_config, train_dataset, eval_dataset)

    print("\nStarting training...\n")
    train_result_metrics = trainer.train()

    print("\nTraining complete.")
    print(f"Final training metrics: {train_result_metrics.metrics}")

    output_dir = Path(training_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving LoRA adapter to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    config_snapshot = {
        "training": asdict(training_cfg),
        "lora": asdict(lora_cfg),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config_snapshot, indent=2), encoding="utf-8"
    )
    print(f"Saved run configuration to {output_dir / 'run_config.json'}")
    print("\nDone. This directory contains a LoRA adapter, not a merged model - "
          "load it with the base model + PeftModel.from_pretrained(...) "
          "(see scripts/test_model.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
