#!/usr/bin/env python3
"""Interactively chat with the base Gemma 3 model or a fine-tuned LoRA adapter.

Usage:
    # Base model only
    python scripts/test_model.py

    # Base model + fine-tuned adapter
    python scripts/test_model.py --adapter outputs/gemma3-1b-corporate-lora

Type 'exit' or 'quit' to end the session.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corporate_chatbot.formatting import build_prompt_messages  # noqa: E402
from corporate_chatbot.model import load_model_for_inference, load_tokenizer  # noqa: E402
from corporate_chatbot.utils import get_hf_token, load_dotenv_if_present, print_environment_report  # noqa: E402

DEFAULT_SYSTEM_PROMPT = (
    "You are the official AI assistant for Technyx Systems. Answer questions about Technyx "
    "Systems using verified information available from the company's official website. Do not "
    "invent or assume information. If the requested information is not available, clearly say "
    "that you do not have verified information about it."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="google/gemma-3-1b-it", help="Base model to load")
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to a trained LoRA adapter directory (omit to test the base model)",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to prepend to the conversation",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the base model in 4-bit (requires a CUDA GPU)",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Don't keep prior turns in context (each question is independent)",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv_if_present()
    args = parse_args()
    print_environment_report()

    hf_token = get_hf_token()

    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = load_tokenizer(args.model_name, hf_token=hf_token)

    label = "fine-tuned adapter" if args.adapter else "base model"
    print(f"Loading model ({label})...")
    model = load_model_for_inference(
        args.model_name,
        adapter_path=args.adapter,
        hf_token=hf_token,
        load_in_4bit=args.load_in_4bit,
    )

    print(f"\nReady. Testing: {label}")
    print("Type 'exit' or 'quit' to end.\n")

    history: list[dict] = []
    while True:
        try:
            user_input = input("You:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        if not user_input:
            continue

        messages = build_prompt_messages(
            user_input,
            system_text=args.system_prompt,
            history=history if not args.no_history else None,
        )
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        import torch

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print(f"\nAssistant:\n> {response}\n")

        if not args.no_history:
            history.append({"role": "user", "content": [{"type": "text", "text": user_input}]})
            history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
