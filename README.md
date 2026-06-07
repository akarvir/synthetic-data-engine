# Synthetic Data Engine

This is a minimalistic CLI for generating LLM training or evaluation datasets. It uses one model as a generator, another model as a judge for generating and filtering good training examples.

## Prerequisites

- An OpenAI compatible language model provider

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

## Development

Run tests:

```bash
uv run pytest
```
