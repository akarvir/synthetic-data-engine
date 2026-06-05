# Synthetic Data Engine

Synthetic Data Engine generates candidate LLM training examples, judges their quality, deduplicates accepted records, and exports JSONL datasets.

## Quick start

```bash
uv run sde run --task tasks/general-instruction.yaml --count 10 --out datasets/general.jsonl
```

The default `local` provider is deterministic and does not require API credentials. It is intended for development, tests, and pipeline smoke checks.

## Commands

```bash
uv run sde generate --task tasks/general-instruction.yaml --count 100
uv run sde judge --run-id <run_id> --min-score 0.8
uv run sde build-dataset --run-id <run_id> --out datasets/general.jsonl
```

The combined `run` command performs all three phases.

## OpenAI-compatible models

Set `OPENAI_API_KEY`, optionally set `OPENAI_BASE_URL`, and provide a model:

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

## Task specs

Task specs are YAML or JSON files. A task defines the desired dataset shape, requirements, topics, difficulty levels, and the output schema used during candidate validation.
