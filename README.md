# Gemma 3 1B Corporate Chatbot - QLoRA Fine-tuning Project

A local, beginner-friendly project for fine-tuning **Google Gemma 3 1B**
(`google/gemma-3-1b-it`) into a corporate website chatbot persona, using
**QLoRA** (4-bit quantization + LoRA adapters).

This project **only covers fine-tuning**. It does not deploy a chatbot, run
a web server, or implement RAG - it produces a trained LoRA adapter and the
tools to validate/test it, so you can plug that into a deployment or a RAG
pipeline later.

---

## 1. Project overview

The goal is to teach a small instruction-tuned model (Gemma 3 1B) how to
*behave* as a corporate website assistant:

- Answer clearly, concisely, and in a professional-but-conversational tone.
- Never invent services, locations, prices, employees, certifications, or
  policies it wasn't actually given.
- Say "I don't know" naturally when it doesn't have verified information,
  instead of guessing or being robotic about it.
- Redirect unrelated/out-of-scope questions politely.

This is done with a small, hand-written example dataset (`data/examples/`)
using a **fictional** company ("Example Technologies") as a template. You
are expected to replace that dataset with your real company's data before
training on anything you intend to actually use.

## 2. Architecture

**Today** (what this project builds):

```
User -> Chatbot UI (not built here) -> Backend (not built here) -> Fine-tuned Gemma 3 1B (LoRA adapter)
```

**Later** (once you add retrieval - see section 12):

```
User -> Backend -> RAG (retriever over your docs) -> Relevant context -> Gemma 3 1B -> Answer
```

The fine-tuned model's job is to be a good *conversational layer* - tone,
formatting, refusal behavior - while a retriever supplies the actual
up-to-date facts. This project is structured (see `src/corporate_chatbot/`)
so that adding a retrieval step later doesn't require rewriting the training
pipeline: the retriever would just inject retrieved context into the
`system` message before generation, using the same `formatting.py` /
`model.py` helpers already used for inference.

## 3. Why Gemma 3 1B

- Small enough to fine-tune and run inference on a single modest GPU
  (or eventually CPU/edge deployment via quantized runtimes like
  llama.cpp/Ollama), which fits a website chatbot's latency/cost needs.
- Instruction-tuned (`-it`) checkpoint already understands chat-style
  interaction, so fine-tuning only needs to *adjust* behavior, not teach
  instruction-following from scratch.
- A 1B model is realistic to fully load, quantize, and iterate on locally
  or in a free Colab session - larger models add cost and iteration time
  without necessarily improving a narrow, FAQ-style assistant.

## 4. Why QLoRA

| Approach | What changes | Memory needed | Notes |
|---|---|---|---|
| Full fine-tuning | Every parameter in the model | Very high (full-precision weights + optimizer states for the whole model) | Overkill for a small behavioral adjustment; expensive and easy to overfit on a small dataset. |
| LoRA | A small number of injected low-rank adapter matrices; base model frozen | Moderate (base model still loaded in full/half precision) | Much cheaper than full fine-tuning; base weights never change. |
| **QLoRA (used here)** | Same as LoRA, but the frozen base model is loaded in **4-bit** | Low | LoRA's efficiency plus 4-bit quantization of the frozen base model, so training fits on a single consumer/Colab GPU. |

In short: the base Gemma 3 1B weights are frozen and quantized to 4-bit
(via `bitsandbytes`); only small LoRA adapter matrices are trained. Output is
a small adapter (a few tens of MB), not a new copy of the full model.

## 5. Directory structure

```
gemma-corporate-chatbot/
├── README.md                     <- this file
├── LICENSE
├── .gitignore
├── .env.example                  <- copy to .env, fill in HF_TOKEN
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   ├── training.yaml              <- training hyperparameters
│   └── lora.yaml                  <- LoRA hyperparameters
│
├── data/
│   ├── raw/                       <- put your real raw JSONL data here
│   ├── processed/                 <- output of prepare_dataset.py
│   ├── train/                     <- output of split_dataset.py
│   ├── validation/                <- output of split_dataset.py
│   └── examples/
│       ├── corporate_examples.jsonl   <- DEMO DATA (fictional company)
│       └── README.md                  <- dataset format spec
│
├── scripts/                       <- CLI entry points (thin wrappers)
│   ├── prepare_dataset.py
│   ├── validate_dataset.py
│   ├── split_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── test_model.py
│
├── src/corporate_chatbot/         <- actual implementation, importable
│   ├── config.py                  <- YAML config loading (typed)
│   ├── dataset.py                 <- validation, stats, splitting
│   ├── formatting.py              <- messages <-> Gemma chat template
│   ├── model.py                   <- tokenizer/model/quantization/LoRA
│   ├── training.py                <- TRL SFTTrainer construction
│   └── utils.py                   <- hardware detection, JSONL I/O, seeding
│
├── evaluation/
│   ├── test_cases.jsonl           <- base vs fine-tuned comparison prompts
│   └── README.md
│
├── outputs/                        <- trained LoRA adapters land here
└── notebooks/
    └── 01_train_gemma3_1b_qlora.ipynb   <- Colab-friendly wrapper around scripts/
```

## 6. Installation

```bash
git clone <your-fork-or-repo-url>
cd gemma-corporate-chatbot

python -m venv .venv
```

Activate the virtual environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd.exe)
.venv\Scripts\activate.bat
```

Install dependencies:

```bash
pip install -r requirements.txt
```

`torch` and `bitsandbytes` are GPU-dependent packages; on machines without a
compatible NVIDIA GPU/CUDA setup you can still install everything to run
dataset validation/preparation, but actual training requires a CUDA GPU
(see section 15, Hardware).

## 7. Hugging Face access

`google/gemma-3-1b-it` is a **gated** model on Hugging Face. Before you can
download it:

1. Create a Hugging Face account if you don't have one.
2. Visit https://huggingface.co/google/gemma-3-1b-it and accept Google's
   Gemma license/usage terms.
3. Create an access token at https://huggingface.co/settings/tokens
   (a "Read" token is enough).
4. Authenticate one of two ways:
   - **Environment variable**: copy `.env.example` to `.env` and set
     `HF_TOKEN=<your token>`. Never commit `.env` (it's already in
     `.gitignore`).
   - **Hugging Face CLI**: run `huggingface-cli login` and paste your token
     when prompted.

Scripts in this project read `HF_TOKEN` (or the standard
`HUGGING_FACE_HUB_TOKEN`) from the environment automatically via
`python-dotenv`. The token is never printed or logged.

## 8. Dataset preparation

### 8a. Real Technyx Systems dataset (already generated)

`configs/training.yaml` is currently pointed at `data/datasets/train.jsonl`
and `data/datasets/validation.jsonl` - a real, grounded dataset generated by
crawling the public [technyxsystems.com](https://technyxsystems.com/)
website (not the fictional demo data described below). Every question/answer
pair traces back to a specific URL, section, and excerpt; nothing about
Technyx is invented. See **`data/website/README.md`** for the full
crawl -> clean -> extract-facts -> generate-QA -> validate -> split pipeline
(`scripts/webdata/*.py`) and how to re-run it if the site changes.

### 8b. Generic / fictional demo dataset

The rest of this section (and `data/examples/`) describes the original,
smaller, hand-written demo dataset for a fictional company ("Example
Technologies"), useful as a template for adapting this project to a
*different* company later. Put your real company training data in
`data/raw/` as one or more JSONL files, following the exact format
documented in
`data/examples/README.md` (a `messages` list per line, with `system`,
`user`, and `assistant` turns). Start from a copy of
`data/examples/corporate_examples.jsonl` and replace the fictional facts.

**Important dataset principle:** fine-tuning here is meant to teach *tone,
response style, instruction-following, and refusal/unknown-answer behavior*
- not to memorize a large, frequently-changing set of facts. For example:

```
BAD:   Fine-tune hundreds of changing company facts directly into the model.
BETTER: User -> Retriever/RAG -> Relevant company info -> Gemma 3 1B -> Answer
```

Keep the fine-tuning dataset focused on *how* the assistant should behave;
plan to hand it fast-changing facts through retrieval later (section 12).

## 9. Validation

```bash
python scripts/validate_dataset.py --input data/examples/corporate_examples.jsonl
```

This checks JSON syntax, required fields, valid roles, and non-empty
messages, then prints dataset statistics (example count, average
user/assistant message length, role counts). It reports every invalid
record with a specific reason and exits non-zero if anything failed - it
never silently drops bad records.

## 10. Prepare + split

```bash
python scripts/prepare_dataset.py \
    --input data/examples/corporate_examples.jsonl \
    --output data/processed/dataset.jsonl

python scripts/split_dataset.py \
    --input data/processed/dataset.jsonl \
    --output-dir data
```

This writes `data/train/train.jsonl` and `data/validation/validation.jsonl`
using a deterministic seed (default: 90% train / 10% validation, both
configurable via `--val-ratio` and `--seed`). If your dataset is small, the
script prints a warning rather than pretending the validation split is
statistically meaningful.

## 11. Training

```bash
python scripts/train.py --config configs/training.yaml --lora-config configs/lora.yaml
```

This will:

1. Print an environment report (Python/PyTorch/Transformers/CUDA/GPU/VRAM).
2. Load and print the training + LoRA configuration.
3. Validate the train/validation JSONL files.
4. Load the tokenizer and the base model in 4-bit (QLoRA).
5. Prepare the model for k-bit training and attach LoRA adapters (base
   model stays frozen; only adapters are trainable).
6. Train with TRL's `SFTTrainer`, using the tokenizer's own chat template
   to format each conversation (no hand-built special tokens).
7. Save the LoRA adapter (not a merged model), tokenizer files, and a
   snapshot of the exact config used, to `outputs/gemma3-1b-corporate-lora/`.

**Requires a CUDA GPU.** If none is detected, the script exits with an
explanation instead of attempting an impractical CPU run.

### Merging the adapter (not done automatically)

This project intentionally keeps the LoRA adapter separate from the base
model rather than merging automatically. When you're ready to merge (e.g.
for a deployment runtime that wants a single set of weights), you can do it
in a short standalone script using `peft`:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
merged = PeftModel.from_pretrained(base, "outputs/gemma3-1b-corporate-lora").merge_and_unload()
merged.save_pretrained("outputs/gemma3-1b-corporate-merged")
AutoTokenizer.from_pretrained("google/gemma-3-1b-it").save_pretrained("outputs/gemma3-1b-corporate-merged")
```

This project doesn't ship that as a `scripts/` entry point yet since merging
is a deployment-phase decision, not a training-phase one - add it when you
actually need it.

## 12. Testing (interactive)

```bash
python scripts/test_model.py --adapter outputs/gemma3-1b-corporate-lora
```

```
You:
> What services do you provide?

Assistant:
> ...
```

Type `exit` or `quit` to end the session. Omit `--adapter` to chat with the
unmodified base model instead. Add `--load-in-4bit` to load the base model
quantized for inference too (requires a CUDA GPU).

## 13. Evaluation

```bash
python scripts/evaluate.py --adapter outputs/gemma3-1b-corporate-lora
```

Runs `evaluation/test_cases.jsonl` through both the **base model** and the
**fine-tuned model** and prints responses side by side, grouped by category
(service questions, location questions, hallucination resistance, unrelated
questions, greetings, follow-ups, tone, conciseness). This is a qualitative
comparison to read yourself, not an automated score - see
`evaluation/README.md`.

## 14. Output

`outputs/gemma3-1b-corporate-lora/` (after training) contains:

- `adapter_config.json`, `adapter_model.safetensors` - the trained LoRA
  adapter weights (small, a few tens of MB; the frozen base model is *not*
  duplicated here).
- Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, etc.).
- `run_config.json` - a snapshot of the exact training + LoRA config used
  for that run, for reproducibility.
- Trainer checkpoints/logs under the same directory, depending on
  `save_steps`/`save_total_limit` in `configs/training.yaml`.

To use the adapter, load the base model and attach it with
`PeftModel.from_pretrained(base_model, "outputs/gemma3-1b-corporate-lora")`
(exactly what `scripts/test_model.py` and `scripts/evaluate.py` do).

## 15. Hardware

- **Training requires a CUDA-capable NVIDIA GPU.** QLoRA's 4-bit
  quantization (via `bitsandbytes`) targets GPU execution; CPU-only training
  is not practical and this project will tell you so rather than attempting
  it.
- Defaults in `configs/training.yaml` (batch size 2, gradient accumulation
  8, `gradient_checkpointing: true`, `max_seq_length: 1024`) are chosen to
  fit a single ~16GB GPU (e.g. a free/standard Google Colab T4). Increase
  batch size or sequence length only if you've confirmed more VRAM is
  available.
- Dataset validation, preparation, and splitting (`scripts/validate_dataset.py`,
  `scripts/prepare_dataset.py` without `--apply-chat-template`,
  `scripts/split_dataset.py`) have no GPU/model dependency and run anywhere
  Python does.

## 16. Deploying later (not implemented here)

This project stops at "trained adapter + evaluation." Deployment is a
separate phase; once you're ready, possible directions include:

- **llama.cpp** or **Ollama** - convert/quantize a merged model (GGUF) for
  lightweight CPU/edge inference.
- **vLLM** - if you have GPU hardware in production and want higher
  throughput serving.
- **FastAPI** (or a **Laravel API**, or any backend) wrapping model
  inference behind an HTTP endpoint for your website's chat widget.
- **Docker** to package the inference service once it exists.

None of this is implemented yet, intentionally - it's out of scope for a
fine-tuning project.

## 17. RAG preparation (future work, not implemented here)

To keep the chatbot's factual knowledge current without repeated
fine-tuning, the planned future architecture is:

```
Company documents -> Chunking -> Embeddings -> Vector database -> Retriever -> Relevant context -> Gemma 3 1B -> Answer
```

Candidate vector databases for a later phase: **FAISS**, **Chroma**, or
**Qdrant** - pick the simplest one that meets your needs; don't add this
complexity until you actually need dynamic retrieval.

This project is structured to make that addition straightforward without
rewriting the training pipeline:

- `src/corporate_chatbot/formatting.py`'s `build_prompt_messages()` already
  takes a `system_text` argument - a retriever would just inject retrieved
  context there before generation.
- The fine-tuned adapter's job (tone, refusal behavior, formatting) is
  orthogonal to *where the facts come from*, so nothing about the trained
  adapter needs to change when you add retrieval.

## 18. Corporate safety / hallucination behavior

Both the example dataset and the intended fine-tuning behavior emphasize
that the assistant must **never fabricate**:

- services, locations, prices, employees, contact information
- certifications, partnerships, policies, opening hours, guarantees

When information isn't available, the model should respond naturally, e.g.:

> "I don't have verified information about that. Please contact our team
> for confirmation."

...varying the phrasing naturally rather than repeating a single robotic
refusal line (see the variety in `data/examples/corporate_examples.jsonl`).

## 19. What to replace when adding your real company's data

1. `data/examples/corporate_examples.jsonl` (or a new file under
   `data/raw/`) - replace the fictional "Example Technologies" facts and
   system prompt with your real company's verified information and persona.
2. `configs/training.yaml` / `configs/lora.yaml` - tune hyperparameters once
   you know your dataset size and available hardware.
3. `evaluation/test_cases.jsonl` - replace with questions relevant to your
   actual company/domain.
4. The default system prompt in `scripts/test_model.py` and
   `scripts/evaluate.py` (`DEFAULT_SYSTEM_PROMPT`) - update the wording to
   match your real company's name/persona.

## 20. What remains to be done later

- **RAG**: chunking company docs, embeddings, a vector database, and a
  retriever feeding context into the system message (section 17).
- **Deployment**: an inference backend (FastAPI/Laravel/etc.), optional
  merging + quantized export (llama.cpp/Ollama/vLLM), and integrating that
  backend into an actual website chat widget (section 16).
- **More data**: the bundled example dataset is a small demo set (~20
  examples); a real deployment should have a broader, reviewed dataset
  covering your company's actual FAQ surface and edge cases.

---

## Quick command reference

```bash
# Setup
python -m venv .venv && source .venv/bin/activate   # or Windows equivalent above
pip install -r requirements.txt
huggingface-cli login                                # or set HF_TOKEN in .env

# Dataset
python scripts/validate_dataset.py --input data/examples/corporate_examples.jsonl
python scripts/prepare_dataset.py --input data/examples/corporate_examples.jsonl --output data/processed/dataset.jsonl
python scripts/split_dataset.py --input data/processed/dataset.jsonl --output-dir data

# Train
python scripts/train.py --config configs/training.yaml --lora-config configs/lora.yaml

# Test / evaluate
python scripts/test_model.py --adapter outputs/gemma3-1b-corporate-lora
python scripts/evaluate.py --adapter outputs/gemma3-1b-corporate-lora
```
