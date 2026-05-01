# `_batch_runs`

Apply the ARA analyst to every PDF in a folder (or zip archive) under a fixed `(model, temperature)`.

This is the production / single-pass counterpart to `src/_experiments/consistency_check.py`, which runs a full `model x temperature x paper x sample` sweep. Use `run_dataset.py` when you just want one `PaperProfile` per PDF — e.g. to score a whole benchmark dataset with a single chosen configuration.

`--input-dir` may be either a directory of PDFs or a `.zip` archive. Zip inputs are extracted to a temporary directory for the duration of the batch and removed automatically when the run finishes (success, error, or `Ctrl-C` via the `finally` block).

## What it does

For every PDF matched recursively by `--pattern` (default `*.pdf`) under `--input-dir`:

1. Calls `GeminiPaperAnalyst` or `GPTPaperAnalyst` (auto-routed by model name prefix) with the given temperature.
2. Writes the resulting `PaperProfile` JSON to `--output-dir/<paper-stem>_<temp>_<model>.profile.json`.
3. Appends a row to `--output-dir/run_timings.csv` with the columns:
   `timestamp_utc, paper, model, temperature, status, duration_seconds, output_path, error`.
4. On failure, writes a `<paper-stem>_<temp>_<model>.profile.err.txt` sidecar so one bad PDF does not abort the batch.

It resumes by default: an existing `.profile.json` is recorded as `skip` in the CSV and not re-run, while papers that previously errored (only an `.err.txt` sidecar, no `.json`) are picked up automatically. Pass `--force` to re-run successes too, or `--retry-errors` to *only* visit the previously-failed papers.

## Prerequisites

- Environment set up via `uv sync` from `src/`.
- The relevant API key in `src/ara_pipeline/.env` or the process environment:
  - `GEMINI_API_KEY` / `GOOGLE_API_KEY` for Gemini models
  - `GPT_API_KEY` / `OPENAI_API_KEY` for GPT / o-series models
- The `--input-dir` actually contains PDFs. The benchmark folders under `data/` ship with `.placeholder` files and require `uv run python _preparation/download_pdfs.py` first.
- If you point `--input-dir` at a `.zip` under `data/`, the archive must be the real file, not a git-LFS pointer. A 134-byte zip whose first line reads `version https://git-lfs.github.com/spec/v1` is a pointer — fetch the real archive with e.g. `git lfs pull --include='data/resciencec/resciencec_papers_original.zip'`.

## Usage

Run from `src/`:

```bash
uv run python _batch_runs/run_dataset.py \
    --model gemini-2.5-flash \
    --temperature 0.0 \
    --input-dir  ../data/resciencec/papers_original \
    --output-dir ../data/experiments/batch_runs/rescience_c_flash_T0
```

Or feed a zip archive directly (extracted to tmp, deleted when done):

```bash
uv run python _batch_runs/run_dataset.py \
    --model gemini-2.5-flash \
    --temperature 0.0 \
    --input-dir  ../data/resciencec/resciencec_papers_original.zip \
    --output-dir ../data/experiments/batch_runs/rescience_c_flash_T0
```

Re-run only the papers that errored on a previous batch (skips the sweep over successful ones — useful when only a handful of large PDFs failed):

```bash
uv run python _batch_runs/run_dataset.py \
    --model gemini-3.1-pro-preview \
    --temperature 0.0 \
    --workers 30 \
    --input-dir  ../data/resciencec/resciencec_papers_original.zip \
    --output-dir ../data/experiments/batch_runs/rescience_c_gemini3_1_pro_T0 \
    --retry-errors
```

Re-run everything (ignore existing JSONs):

```bash
uv run python _batch_runs/run_dataset.py ... --force
```

### CLI flags

| Flag             | Required | Default   | Notes |
|------------------|----------|-----------|-------|
| `--model`        | yes      | —         | Single model id. `gpt-*` / `o1*` / `o3*` / `o4*` route to OpenAI; everything else routes to Gemini. Reasoning models (`gpt-5*`, o-series) ignore `--temperature` but the value is still recorded in the CSV. |
| `--temperature`  | yes      | —         | Single float. Sane ranges: Gemini & OpenAI chat `[0, 2]`, OSS `[0, 1]`. |
| `--input-dir`    | yes      | —         | Folder containing PDFs OR a `.zip` archive of PDFs (extracted to a tmp dir and removed when the batch ends). |
| `--output-dir`   | yes      | —         | Created if missing. Holds `*.profile.json`, `*.err.txt`, and `run_timings.csv`. |
| `--pattern`      | no       | `*.pdf`   | Glob applied recursively under `--input-dir`. |
| `--workers`      | no       | `8`       | Total parallel API calls. |
| `--gpt-workers`  | no       | `2`       | Sub-cap on concurrent OpenAI calls; tighten for low-tier OpenAI accounts to avoid 429s. |
| `--force`        | no       | off       | Re-run even if the output JSON already exists. |
| `--retry-errors` | no       | off       | Only run papers with a `.profile.err.txt` sidecar in `--output-dir` for this `(model, temperature)`. Implies `--force`. |

## Output layout

```
<output-dir>/
├── 2017_04_article_T0_gemini-2_5-flash.profile.json
├── 2017_04_article_T0_gemini-2_5-flash.profile.err.txt   # only if that run failed
├── 2020_26_article_T0_gemini-2_5-flash.profile.json
├── ...
└── run_timings.csv
```

Filename slug rules (shared with `consistency_check.py`):

- `slug_temp(0.5) -> "T0p5"`, `slug_temp(1.0) -> "T1"`
- `slug_model("gemini-2.5-flash") -> "gemini-2_5-flash"` (dots become underscores; reversible)

## Re-running and recovery

- **Skip done work**: just re-invoke the same command. Existing JSONs are skipped (recorded as `status=skip` in the CSV); papers without a JSON (including ones that previously errored) are automatically picked up.
- **Retry only failures (targeted)**: pass `--retry-errors`. The script scans `--output-dir` for `*.profile.err.txt` sidecars matching the given `(model, temperature)`, builds runs only for those papers, and forces re-execution. Exits early with a message if no sidecars are found. Cheaper than a plain re-run when there are many successful papers, since it avoids the file-existence sweep over the rest of the dataset.
- **Force a clean re-run**: pass `--force`. Successful runs are overwritten; stale `.err.txt` sidecars are removed on a successful overwrite.

## Downstream

Profile JSONs follow the canonical `PaperProfile` schema defined in `src/ara_pipeline/online_llm_pipelines.py` and can be fed directly into:

- `ara_pipeline.repro_scoring` to compute the reproducibility index.
- `src/_analysis/workflow_graph_analysis.ipynb` for per-paper inspection (set `JSON_PATH` to one of the produced files).
