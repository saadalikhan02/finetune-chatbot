# Evaluation

`test_cases.jsonl` is a small, hand-written set of prompts used by
`scripts/evaluate.py` to qualitatively compare the **base model** against
the **fine-tuned model** on the same questions.

## What this is

- A readable side-by-side comparison of generated responses, grouped by
  category (service questions, location questions, hallucination
  resistance, unrelated questions, greetings, follow-ups, tone, etc.).
- A quick way to sanity-check whether fine-tuning nudged the model toward
  the desired behavior (concise, professional, refuses to invent facts).

## What this is NOT

- Not an automated or scientific benchmark - there's no scoring, and no
  ground-truth "correct" answer is checked programmatically.
- Not a substitute for reading the actual generated text yourself.
- Not statistically meaningful with this few examples - it's a spot check,
  not a metric to optimize against.

## Format

Each line is a JSON object:

```json
{"category": "service_questions", "user_input": "What services do you offer?"}
```

Optional fields:

- `system_prompt`: override the default system prompt for this case.
- `history`: prior turns (same `messages` shape as the training data) to
  test follow-up-question behavior.

## Running it

```bash
python scripts/evaluate.py --adapter outputs/gemma3-1b-corporate-lora
```

Add `--skip-base` to only generate from the fine-tuned model (faster, if you
just want to see current behavior rather than a comparison).
