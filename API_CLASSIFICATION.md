# OpenAI-compatible API classification

The API runner adapts the source project's frozen-model API workflow to the prepared JSON packages in `data_json/four_batches`. It supports DeepSeek and other OpenAI-compatible chat-completions endpoints.

## Data boundary and request controls

- Preview mode is the default. No request is sent unless `--execute` is present.
- Live mode also requires a positive `--max-requests`, and the selected experiment count must fit within that budget.
- One request predicts both Val and Test for one experiment. Requests use temperature 0 by default, disable SDK retries, and are cached by endpoint, model, messages, and generation parameters.
- The inference path reads only `prompt.txt`, `train.json`, `val.json`, and `test.json` from each package, plus `index_current.csv` for package selection. It does not read `val_labels.json`, `test_labels.json`, `metadata.json`, or private mappings. Scoring is separate.
- The web prompts contain context-only instructions that prohibit API use. The API adapter removes that execution-only restriction and replaces the context-only phrasing; it preserves the task definition, permitted labels, coverage requirements, and output schema. API results are a separate experimental condition from context-only web predictions.
- API keys are loaded from the ignored local `local_api_config.py`. Never place credentials in configuration committed to version control, command-line arguments, or result files.

## Setup

```powershell
python -m pip install -r requirements-api.txt
Copy-Item local_api_config.example.py local_api_config.py
```

Set the provider's exact model ID, base URL, and key in `local_api_config.py`. The template contains `deepseek` and `openai_compatible` entries; use `--provider` to select a key.

## Preview and run

Preview the first experiment in T1. This prints the selected package and whether local provider settings are complete, without making an API call:

```powershell
python scripts/run_api_classifier.py --batch batch_01_T1 --limit 1
```

Then explicitly allow one request:

```powershell
python scripts/run_api_classifier.py --batch batch_01_T1 --limit 1 --execute --max-requests 1
```

Select a package by exact ID using `--experiment-id <id>`, or select more experiments with `--limit N` and set `--max-requests` to at least N. A matching cached response does not consume a new-request budget. Raise `--max-tokens` for packages with larger Val/Test partitions.

## Outputs

- `outputs/api_classification/predictions.jsonl`: append-only status and output-path ledger.
- `outputs/api_classification/experiments/<experiment_id>/predictions.json`: validated, frozen predictions with output issues listed.
- `outputs/api_classification/experiments/<experiment_id>/raw_response.txt`: original model response for audit.
- `outputs/api_classification/cache/<request-sha256>.json`: cached provider response.

The runner validates experiment identity, duplicate/missing/extra IDs, and candidate labels from the package prompt. It does not score predictions against ground truth. Score only after predictions have been saved and frozen.
