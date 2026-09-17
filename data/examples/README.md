# Example dataset (DEMO DATA)

`corporate_examples.jsonl` in this folder is **demo data** for a fictional
company called "Example Technologies". It exists to show the dataset format
and the kind of chatbot *behavior* this project is meant to teach. It is not
real company information, and it is not meant to be used as-is for a real
deployment.

## Format

Each line is one JSON object with a `messages` array, matching the format
Hugging Face's chat templates expect:

```json
{
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "..."}]},
    {"role": "user", "content": [{"type": "text", "text": "..."}]},
    {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
  ]
}
```

Notes:

- `content` may be a plain string (`"content": "..."`) or a list of text
  parts (`[{"type": "text", "text": "..."}]`). This project's code accepts
  both, but the list form is used consistently in the example data since it
  matches Gemma 3's chat template and leaves room for future multimodal
  content types if ever needed.
- Valid roles are `system`, `user`, and `assistant`.
- A conversation should start with a `system` or `user` message and must
  contain at least one `user` and one `assistant` message.
- Multi-turn examples (a follow-up question) are supported - just add more
  `user`/`assistant` pairs to the `messages` list.

## What this data is teaching

Per the project's core principle (see the main README's "Dataset Principle"
section), these examples exist to teach **tone, behavior, and formatting**,
not a large body of memorized facts:

- Answer clearly and concisely, in a professional-but-conversational tone.
- Never invent services, locations, prices, employees, or policies.
- When information isn't verified/available, say so naturally instead of
  guessing or refusing robotically.
- Politely redirect out-of-scope or unrelated questions back to what the
  assistant can actually help with.
- Handle greetings, thanks, ambiguous questions, and follow-up questions
  naturally.

## Replacing this with your own data

When you're ready to fine-tune on a real company:

1. Copy this file's structure into `data/raw/` (e.g. `data/raw/company_data.jsonl`).
2. Replace the fictional facts and system prompt with your real company's
   verified information and desired persona.
3. Keep the same mix of categories (services, locations, contact info,
   unknown/refusal behavior, greetings, follow-ups, etc.) - the *behavior*
   coverage matters more than raw example count.
4. Run it through `scripts/validate_dataset.py`, then
   `scripts/prepare_dataset.py`, then `scripts/split_dataset.py`, exactly as
   described in the main README.

Remember: facts that change often (prices, staff, office addresses) are
better served by adding retrieval (RAG) later rather than by fine-tuning
them directly into the model - see the main README's RAG section.
