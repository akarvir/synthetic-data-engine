# Synthetic Data Engine

Synthetic Data Engine is a CLI for generating LLM training or evaluation datasets. It uses one model as a generator, another model as a judge, stores every run in SQLite, and exports the accepted examples as JSONL with a manifest.

The default workflow is:

```text
task spec -> generator model -> candidates -> judge model -> accepted dataset
```

## What You Can Use It For

- Generate instruction-answer datasets for fine-tuning or evaluation.
- Build small domain-specific datasets from repeatable YAML task specs.
- Compare generator and judge model behavior across local and hosted models.
- Audit synthetic examples with stored prompts, scores, verdicts, and run metadata.
- Iterate locally with Ollama before scaling to another model server.

## Prerequisites

- Python managed through `uv`.
- At least one model provider:
  - Ollama for free local models.
  - Any OpenAI-compatible `/v1/chat/completions` server.
  - The built-in `local` deterministic provider for smoke tests only.

Install/sync the project:

```bash
uv sync
```

## Quick Start: Local Ollama

Start Ollama and pull a model:

```bash
ollama pull llama3.1
ollama serve
```

Generate, judge, and export a dataset:

```bash
uv run sde run \
  --task tasks/general-instruction.yaml \
  --generator-provider ollama \
  --generator-model llama3.1 \
  --judge-provider ollama \
  --judge-model llama3.1 \
  --count 10 \
  --out datasets/ollama.jsonl
```

Expected outputs:

```text
datasets/ollama.jsonl
datasets/ollama.jsonl.manifest.json
runs/sde.sqlite
```

If Ollama is not running at `http://localhost:11434/v1`, set:

```bash
export OLLAMA_BASE_URL=http://localhost:11434/v1
```

## Quick Start: Other OpenAI-Compatible Servers

Use this for hosted OpenAI-compatible APIs or local servers that expose `/v1/chat/completions`:

```bash
export OPENAI_API_KEY=<your-key>
export OPENAI_BASE_URL=https://api.openai.com/v1
```

```bash
uv run sde run \
  --task tasks/general-instruction.yaml \
  --generator-provider openai-compatible \
  --generator-model gpt-4.1-mini \
  --judge-provider openai-compatible \
  --judge-model gpt-4.1 \
  --count 50 \
  --out datasets/general.jsonl
```

## CLI Commands

Run the full pipeline:

```bash
uv run sde run --task tasks/general-instruction.yaml --count 10 --out datasets/general.jsonl
```

Run the phases separately:

```bash
uv run sde generate --task tasks/general-instruction.yaml --count 100
uv run sde judge --run-id latest
uv run sde build-dataset --run-id latest --out datasets/general.jsonl
```

Inspect and operate on runs:

```bash
uv run sde validate-task --task tasks/general-instruction.yaml
uv run sde list-runs
uv run sde report --run-id latest
uv run sde list-failures --run-id latest
uv run sde inspect --run-id latest --max-items 3
uv run sde show-candidate --candidate-id <candidate_id>
```

Resume or top up a run:

```bash
uv run sde generate --run-id latest --target-count 100
```

## Task Specs

Task specs are YAML or JSON files. A task defines the dataset shape, generation requirements, topics, difficulty levels, selection defaults, and output schema.

Example:

```yaml
name: general-instruction
domain: reasoning
description: Generate instruction-answer examples that teach clear reasoning.
requirements:
  - The prompt must be specific and answerable.
  - The answer must be concrete and useful.
output_schema:
  type: object
  required:
    - prompt
    - answer
  properties:
    prompt:
      type: string
    answer:
      type: string
    metadata:
      type: object
selection:
  min_score: 0.8
```

The included starter task is [tasks/general-instruction.yaml](tasks/general-instruction.yaml).

## Data And Audit Trail

By default the SQLite database is:

```text
runs/sde.sqlite
```

The database stores:

- runs
- generated candidates
- judge results
- model call failures
- generation and judge prompt messages

Every dataset export writes:

- a JSONL dataset file
- a sibling `.manifest.json` file with run ID, row count, selection settings, and run report

Disable manifest creation with:

```bash
uv run sde build-dataset --run-id latest --out datasets/general.jsonl --no-manifest
```

## Selection Defaults

When omitted, commands use selection settings from the task spec:

- `selection.min_score`, defaulting to `0.8`
- `selection.max_items`, when present
- `selection.difficulty_distribution`, when present

You can override common settings from the CLI:

```bash
uv run sde build-dataset --run-id latest --min-score 0.85 --max-items 100
```

## Development

Run tests:

```bash
uv run pytest
```
