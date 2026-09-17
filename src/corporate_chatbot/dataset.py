"""Loading, validating, and splitting the corporate chatbot JSONL dataset.

This module deliberately does not know anything about the actual company
data - it only understands the generic ``{"messages": [...]}}`` schema
described in data/examples/README.md.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .formatting import VALID_ROLES, extract_text
from .utils import read_jsonl


@dataclass
class ValidationIssue:
    index: int
    message: str

    def __str__(self) -> str:
        return f"record {self.index}: {self.message}"


@dataclass
class ValidationResult:
    valid_records: list[dict[str, Any]] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.issues) == 0

    @property
    def total(self) -> int:
        return len(self.valid_records) + len(
            {issue.index for issue in self.issues}
        )


def validate_record(record: Any, index: int) -> list[str]:
    """Validate a single parsed JSON record. Returns a list of human-readable
    error strings; empty means the record is valid."""
    errors: list[str] = []

    if not isinstance(record, dict):
        return [f"expected a JSON object, got {type(record).__name__}"]

    if "messages" not in record:
        return ["missing required field 'messages'"]

    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        return ["'messages' must be a non-empty list"]

    seen_roles = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"messages[{i}] must be an object")
            continue

        role = message.get("role")
        if role is None:
            errors.append(f"messages[{i}] missing 'role'")
        elif role not in VALID_ROLES:
            errors.append(
                f"messages[{i}] has invalid role '{role}' "
                f"(expected one of {sorted(VALID_ROLES)})"
            )
        else:
            seen_roles.append(role)

        if "content" not in message:
            errors.append(f"messages[{i}] missing 'content'")
            continue

        try:
            text = extract_text(message["content"])
        except ValueError as e:
            errors.append(f"messages[{i}] {e}")
            continue

        if not text.strip():
            errors.append(f"messages[{i}] ('{role}') has empty content")

    if "assistant" not in seen_roles:
        errors.append("conversation has no 'assistant' message")
    if "user" not in seen_roles:
        errors.append("conversation has no 'user' message")

    if seen_roles and seen_roles[0] not in ("system", "user"):
        errors.append("conversation must start with a 'system' or 'user' message")

    return errors


def load_and_validate(path: str | Path) -> ValidationResult:
    """Load a JSONL file and validate every record.

    Invalid JSON lines are reported as issues (not raised), so a single
    malformed record does not abort validation of the rest of the file
    where possible. Fully unparseable files still raise, since there is
    nothing to iterate over.
    """
    path = Path(path)
    result = ValidationResult()

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(raw_lines):
        line = line.strip()
        if not line:
            continue
        import json

        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            result.issues.append(ValidationIssue(index, f"invalid JSON ({e.msg})"))
            continue

        errors = validate_record(record, index)
        if errors:
            for error in errors:
                result.issues.append(ValidationIssue(index, error))
        else:
            result.valid_records.append(record)

    return result


def compute_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute simple, honest descriptive statistics over valid records."""
    user_lengths = []
    assistant_lengths = []
    role_counts: dict[str, int] = {}

    for record in records:
        for message in record["messages"]:
            role = message["role"]
            role_counts[role] = role_counts.get(role, 0) + 1
            text = extract_text(message["content"])
            if role == "user":
                user_lengths.append(len(text))
            elif role == "assistant":
                assistant_lengths.append(len(text))

    def avg(values: list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "num_examples": len(records),
        "role_counts": role_counts,
        "avg_user_message_chars": round(avg(user_lengths), 1),
        "avg_assistant_message_chars": round(avg(assistant_lengths), 1),
        "num_user_messages": len(user_lengths),
        "num_assistant_messages": len(assistant_lengths),
    }


def split_records(
    records: list[dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically shuffle and split records into (train, validation).

    Does not pretend a tiny validation split is statistically meaningful -
    callers (scripts/split_dataset.py) print a warning for small datasets.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    shuffled = records.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    num_val = max(1, round(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_records = shuffled[:num_val]
    train_records = shuffled[num_val:]
    return train_records, val_records


def to_hf_dataset(records: list[dict[str, Any]]):
    """Convert a list of ``{"messages": [...]}}`` dicts into a
    ``datasets.Dataset`` while preserving the messages structure."""
    from datasets import Dataset

    return Dataset.from_list(records)
