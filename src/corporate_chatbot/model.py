"""Model/tokenizer loading, quantization, and LoRA setup for Gemma 3.

Keeps all Gemma-specific and PEFT/bitsandbytes-specific wiring in one place
so scripts/train.py, scripts/evaluate.py, and scripts/test_model.py share the
exact same loading logic.
"""

from __future__ import annotations

from typing import Any

from .config import LoraSettings, TrainingSettings


def resolve_compute_dtype(setting: str, cuda_available: bool):
    """Resolve the 'bnb_4bit_compute_dtype' config value ('auto', 'bfloat16',
    'float16', 'float32') to an actual torch dtype."""
    import torch

    if setting == "auto":
        if cuda_available and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if setting not in mapping:
        raise ValueError(
            f"Unsupported bnb_4bit_compute_dtype '{setting}'. "
            f"Expected one of: 'auto', {sorted(mapping)}"
        )
    return mapping[setting]


def build_bnb_config(training_cfg: TrainingSettings, cuda_available: bool):
    """Build a BitsAndBytesConfig for 4-bit QLoRA, or return None if 4-bit
    loading is disabled or no GPU is available (bitsandbytes 4-bit
    quantization requires CUDA)."""
    if not training_cfg.load_in_4bit:
        return None
    if not cuda_available:
        return None

    from transformers import BitsAndBytesConfig

    compute_dtype = resolve_compute_dtype(training_cfg.bnb_4bit_compute_dtype, cuda_available)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=training_cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=training_cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_tokenizer(model_name: str, hf_token: str | None = None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(
    model_name: str,
    bnb_config: Any = None,
    hf_token: str | None = None,
    device_map: str | None = "auto",
):
    """Load the base Gemma 3 model, quantized to 4-bit when a BitsAndBytesConfig
    is provided, otherwise in full precision (e.g. for CPU-only smoke tests)."""
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"token": hf_token}
    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config
        kwargs["device_map"] = device_map
    else:
        kwargs["torch_dtype"] = torch.float32

    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def build_lora_config(lora_cfg: LoraSettings):
    from peft import LoraConfig, TaskType

    task_type = getattr(TaskType, lora_cfg.task_type)
    return LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        target_modules=list(lora_cfg.target_modules),
        task_type=task_type,
    )


def prepare_model_for_qlora_training(model: Any, gradient_checkpointing: bool = True):
    """Prepare a quantized model for k-bit training (casts norms to fp32,
    enables input gradients, optionally enables gradient checkpointing)."""
    from peft import prepare_model_for_kbit_training

    return prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing
    )


def load_model_for_inference(
    model_name: str,
    adapter_path: str | None = None,
    hf_token: str | None = None,
    load_in_4bit: bool = False,
):
    """Load the base model (optionally 4-bit quantized) and, if given, attach
    a trained LoRA adapter on top. Used by scripts/test_model.py and
    scripts/evaluate.py to load either the base or fine-tuned model.
    """
    import torch

    cuda_available = torch.cuda.is_available()
    bnb_config = None
    if load_in_4bit and cuda_available:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=resolve_compute_dtype("auto", cuda_available),
        )

    model = load_base_model(
        model_name,
        bnb_config=bnb_config,
        hf_token=hf_token,
        device_map="auto" if cuda_available else None,
    )

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model
