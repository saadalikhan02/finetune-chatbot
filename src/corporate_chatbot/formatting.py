"""Conversion between the project's JSONL message format and the plain text
Gemma 3's tokenizer chat template expects.

Dataset format (see data/examples/README.md for the full spec)::

    {
      "messages": [
        {"role": "system", "content": [{"type": "text", "text": "..."}]},
        {"role": "user", "content": [{"type": "text", "text": "..."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
      ]
    }

``content`` may also be a plain string - both forms are accepted everywhere
in this module. We never hand-build Gemma special tokens (``<start_of_turn>``
etc.) ourselves: whenever a tokenizer is available we delegate templating to
``tokenizer.apply_chat_template``, which already knows the model's turn
format, role names, and how to fold a leading system message into the first
user turn. The plain-text fallback here exists only for tokenizer-free
statistics/validation (e.g. scripts/validate_dataset.py) where no model is
loaded.
"""

from __future__ import annotations

from typing import Any

VALID_ROLES = {"system", "user", "assistant"}


def extract_text(content: Any) -> str:
    """Extract plain text from a message's ``content`` field.

    Accepts either a plain string or the list-of-parts form
    ``[{"type": "text", "text": "..."}]`` used by the project's dataset
    format. Non-text parts (e.g. images) are ignored - this project is
    text-only.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    raise ValueError(f"Unsupported content type: {type(content)!r}")


def messages_to_plain_text(messages: list[dict[str, Any]]) -> str:
    """Render messages as simple ``role: text`` lines, with no model-specific
    special tokens. Used only where a tokenizer isn't loaded (e.g. dataset
    validation stats) - not used to build training inputs.
    """
    lines = []
    for message in messages:
        role = message.get("role", "unknown")
        text = extract_text(message.get("content", ""))
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def apply_gemma_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool = False,
) -> str:
    """Render ``messages`` into the exact text Gemma 3 expects, using the
    tokenizer's own chat template rather than hand-built special tokens.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def format_example_for_training(tokenizer: Any, example: dict[str, Any]) -> dict[str, str]:
    """Convert one dataset record (``{"messages": [...]}}``) into the
    ``{"text": ...}`` form the SFT trainer consumes, using the tokenizer's
    chat template. The full conversation (including the assistant's answer)
    is included, with no trailing generation prompt, since this is a
    complete training target rather than a prompt awaiting a response.
    """
    messages = example["messages"]
    text = apply_gemma_chat_template(tokenizer, messages, add_generation_prompt=False)
    return {"text": text}


def build_prompt_messages(
    user_text: str,
    system_text: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a messages list for inference: optional system message, optional
    prior turns, then the new user turn. Used by scripts/test_model.py and
    scripts/evaluate.py.
    """
    messages: list[dict[str, Any]] = []
    if system_text:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
    return messages
