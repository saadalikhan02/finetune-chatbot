#!/usr/bin/env python3
"""Generate and compare BASE MODEL vs FINE-TUNED MODEL responses on a fixed
set of test cases.

This is a qualitative, side-by-side comparison intended to help you eyeball
whether fine-tuning changed tone/behavior in the intended direction (e.g.
conciseness, refusal behavior on unknown info, resisting hallucination). It
is NOT a scientific or automated correctness benchmark - there is no ground
truth scoring here, just generated text for you to read.

Usage:
    python scripts/evaluate.py \\
        --adapter outputs/gemma3-1b-corporate-lora \\
        --test-cases evaluation/test_cases.jsonl

    # Fine-tuned model only (skip loading the base model twice):
    python scripts/evaluate.py --adapter outputs/gemma3-1b-corporate-lora --skip-base
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corporate_chatbot.formatting import build_prompt_messages  # noqa: E402
from corporate_chatbot.model import load_model_for_inference, load_tokenizer  # noqa: E402
from corporate_chatbot.utils import (  # noqa: E402
    get_hf_token,
    load_dotenv_if_present,
    print_environment_report,
    read_jsonl,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are the official AI assistant for Technyx Systems. Speak naturally as part of the "
    "company (use \"we\"/\"our\"), never as someone describing or citing a website. Answer only "
    "using verified company information. Do not invent or assume information. If the requested "
    "information is not available, say so directly - for example \"I don't have verified "
    "information about that\" - never phrase it as \"the website doesn't say\" or similar."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="google/gemma-3-1b-it")
    parser.add_argument("--adapter", default=None, help="Path to the fine-tuned LoRA adapter")
    parser.add_argument("--test-cases", default="evaluation/test_cases.jsonl", type=Path)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Only run the fine-tuned model (skip the base-model comparison column)",
    )
    return parser.parse_args()


def generate(
    model,
    tokenizer,
    system_prompt: str,
    user_input: str,
    max_new_tokens: int,
    history: list[dict] | None = None,
) -> str:
    import torch

    messages = build_prompt_messages(user_input, system_text=system_prompt, history=history)
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> int:
    load_dotenv_if_present()
    args = parse_args()
    print_environment_report()

    if not args.test_cases.exists():
        print(f"ERROR: test cases file not found: {args.test_cases}", file=sys.stderr)
        return 1

    test_cases = read_jsonl(args.test_cases)
    print(f"\nLoaded {len(test_cases)} test case(s) from {args.test_cases}")

    hf_token = get_hf_token()
    tokenizer = load_tokenizer(args.model_name, hf_token=hf_token)

    base_model = None
    if not args.skip_base:
        print(f"\nLoading BASE model: {args.model_name}")
        base_model = load_model_for_inference(
            args.model_name, adapter_path=None, hf_token=hf_token, load_in_4bit=args.load_in_4bit
        )

    print(f"Loading FINE-TUNED model (adapter: {args.adapter})")
    tuned_model = load_model_for_inference(
        args.model_name, adapter_path=args.adapter, hf_token=hf_token, load_in_4bit=args.load_in_4bit
    )

    print("\n" + "=" * 80)
    for i, case in enumerate(test_cases, start=1):
        category = case.get("category", "uncategorized")
        user_input = case["user_input"]
        system_prompt = case.get("system_prompt", args.system_prompt)
        history = case.get("history")

        print(f"\n[{i}/{len(test_cases)}] category: {category}")
        if history:
            print("(follow-up question; prior turns omitted for brevity)")
        print(f"Q: {user_input}")

        if base_model is not None:
            base_response = generate(
                base_model, tokenizer, system_prompt, user_input, args.max_new_tokens, history=history
            )
            print(f"\nBASE MODEL:\n  {base_response}")

        tuned_response = generate(
            tuned_model, tokenizer, system_prompt, user_input, args.max_new_tokens, history=history
        )
        print(f"\nFINE-TUNED MODEL:\n  {tuned_response}")
        print("\n" + "-" * 80)

    print(
        "\nReminder: this is a qualitative comparison, not an automated score. "
        "Read the responses yourself and judge tone, conciseness, and whether "
        "the model avoided inventing company facts it wasn't trained on."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
