"""Parallel consistency experiment for local Qwen3 models.

Runs every (model, temperature, paper, sample) combination through
VLLMOfflinePaperAnalyst and writes the resulting PaperProfile to
data/experiments/consistency_checks/rescience_c_local_models/<paper>_T<t>_exe<n>_<model>.profile.json
under the repository root.

Usage (from src/):
    uv run python _experiments/consistency_check_offline_models.py --workers 8
    uv run python _experiments/consistency_check_offline_models.py --dry-run
    uv run python _experiments/consistency_check_offline_models.py --models qwen3-32b --workers 4
    uv run python _experiments/consistency_check_offline_models.py --force

Resumes by default: existing output files are skipped unless --force is passed.
Failures land in a sidecar <stem>.err.txt so a single bad run doesn't abort
the sweep.
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Make `ara_pipeline` importable when this script is run directly from `src/`.
# `python file.py` puts the script's directory on sys.path, not its parent;
# inserting `src/` lets `from ara_pipeline... import ...` resolve without an
# editable install (mirrors the shim in consistency_check.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ara_pipeline.local_llm_pipeline import VLLMOfflinePaperAnalyst

# -- experiment grid ----------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_DIR = REPO_ROOT / "data" / "resciencec" / "papers_original"
OUTPUT_DIR = (
    REPO_ROOT / "data" / "experiments" / "consistency_checks" / "rescience_c_local_models"
)

TIMINGS_CSV = OUTPUT_DIR / "run_timings.csv"
TIMINGS_FIELDS = [
    "timestamp_utc",
    "paper",
    "model",
    "temperature",
    "sample",
    "status",
    "duration_seconds",
    "output_path",
    "error",
]
_csv_lock = threading.Lock()


def append_timing(
    csv_path: Path,
    run: Run,
    status: str,
    duration: float,
    error: str = "",
) -> None:
    """Thread-safe append of one execution record to the timings CSV."""
    with _csv_lock:
        new_file = not csv_path.exists()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TIMINGS_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                    "paper": run.paper_pdf.name,
                    "model": run.model,
                    "temperature": run.temperature,
                    "sample": run.sample,
                    "status": status,
                    "duration_seconds": f"{duration:.3f}",
                    "output_path": str(run.out),
                    "error": error,
                }
            )


PAPERS = [
    "2017_04_article.pdf",
    "2020_26_article.pdf",
    "2017_01_article.pdf",
    "2025_03_article.pdf",
    "2017_08_article.pdf",
    "2022_30_article.pdf",
    "2021_34_article.pdf",
    "2023_17_article.pdf",
    "2023_15_article.pdf",
    "2022_38_article.pdf",
]

TEMPS = [0.0, 0.25, 0.5, 0.75, 1.0]

MODELS: dict[str, list[float]] = {
    "qwen3-8b":  TEMPS,   # Qwen/Qwen3-8B   ~16 GB
    "qwen3-32b": TEMPS,   # Qwen/Qwen3-32B  ~64 GB
}

HF_MODEL_IDS: dict[str, str] = {
    "qwen3-8b":  "Qwen/Qwen3-8B",
    "qwen3-32b": "Qwen/Qwen3-32B",
}

# vLLM error message tells us the max supportable length; set it just below that.
VLLM_MAX_MODEL_LEN: dict[str, int] = {
    "qwen3-32b": 28000,
}

N_SAMPLES = 10

# -- naming -------------------------------------------------------------------


def slug_model(model: str) -> str:
    """Filename-safe alias: keep dashes, dots -> underscores. Reversible."""
    return model.replace(".", "_")


def slug_temp(t: float) -> str:
    """T0 / T0p5 / T1 / T1p5 / T2 -- avoids dots, parses cleanly."""
    s = f"{t:.2f}".rstrip("0").rstrip(".")
    return "T" + s.replace(".", "p")


def output_path(paper_pdf: str, t: float, sample: int, model: str) -> Path:
    stem = Path(paper_pdf).stem
    return OUTPUT_DIR / f"{stem}_{slug_temp(t)}_exe{sample}_{slug_model(model)}.profile.json"


# -- run unit -----------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    paper_pdf: Path
    model: str
    temperature: float
    sample: int

    @property
    def out(self) -> Path:
        return output_path(self.paper_pdf.name, self.temperature, self.sample, self.model)


def prewarm_offline_models(model_keys: list[str]) -> None:
    """Initialize vLLM engines before the thread pool starts.

    Fails immediately with a clear message if the GPU is unavailable,
    preventing a flood of identical errors from every queued run.
    """
    seen: set[str] = set()
    for key in model_keys:
        if key in seen:
            continue
        seen.add(key)
        hf_id = HF_MODEL_IDS[key]
        print(f"Loading vLLM engine: {hf_id} ...", flush=True)
        VLLMOfflinePaperAnalyst(
            model=hf_id,
            temperature=0.0,
            max_model_len=VLLM_MAX_MODEL_LEN.get(key),
        )
        print(f"  engine ready: {hf_id}", flush=True)


def execute(run: Run) -> tuple[Run, str, float]:
    t0 = time.monotonic()
    try:
        analyst = VLLMOfflinePaperAnalyst(
            model=HF_MODEL_IDS.get(run.model, run.model),
            temperature=run.temperature,
            max_model_len=VLLM_MAX_MODEL_LEN.get(run.model),
        )
        profile = analyst.analyze(run.paper_pdf, verbose=False)
        run.out.parent.mkdir(parents=True, exist_ok=True)
        run.out.write_text(profile.model_dump_json(indent=2))
        err = run.out.with_suffix(".err.txt")
        if err.exists():
            err.unlink()
        dt = time.monotonic() - t0
        append_timing(TIMINGS_CSV, run, "ok", dt)
        return run, "ok", dt
    except Exception as exc:
        dt = time.monotonic() - t0
        err = run.out.with_suffix(".err.txt")
        err.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(f"{type(exc).__name__}: {exc}")
        status = f"err:{type(exc).__name__}"
        append_timing(TIMINGS_CSV, run, status, dt, error=f"{type(exc).__name__}: {exc}")
        return run, status, dt


def build_runs(
    models: list[str] | None = None,
    papers: list[str] | None = None,
    n_samples: int = N_SAMPLES,
) -> list[Run]:
    selected_models = {m: MODELS[m] for m in (models or list(MODELS))}
    selected_papers = papers or PAPERS
    runs: list[Run] = []
    for paper in selected_papers:
        pdf = PAPER_DIR / paper
        if not pdf.is_file():
            print(f"WARN: missing PDF, skipping: {pdf}", file=sys.stderr)
            continue
        for model, temps in selected_models.items():
            for t in temps:
                for s in range(1, n_samples + 1):
                    runs.append(Run(pdf, model, t, s))
    return runs


# -- main ---------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="parallel worker threads; tune to GPU memory / throughput",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run even if output JSON already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned grid and exit",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS),
        help="restrict to a subset of models (default: all configured)",
    )
    parser.add_argument(
        "--papers",
        nargs="+",
        help="restrict to a subset of paper filenames (default: all 10)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=N_SAMPLES,
        help=f"samples per (model, temp, paper); default {N_SAMPLES}",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = build_runs(args.models, args.papers, args.samples)
    n_models = len({r.model for r in runs})
    n_papers = len({r.paper_pdf for r in runs})
    n_existing = sum(1 for r in runs if r.out.exists())
    n_pending = len(runs) - n_existing
    print(
        f"Plan: {len(runs)} runs  "
        f"({n_models} model(s), {args.samples} samples, {n_papers} paper(s))",
        flush=True,
    )
    print(
        f"      pending={n_pending}  already_done={n_existing}"
        f"{'  (use --force to re-run)' if n_existing and not args.force else ''}",
        flush=True,
    )
    if args.dry_run:
        if not runs:
            print(
                f"  (no runs: check PAPER_DIR={PAPER_DIR} and --papers/--models filters)",
                flush=True,
            )
            return 0
        print("\nPlanned runs:", flush=True)
        for r in runs:
            tag = "exists" if r.out.exists() else "pending"
            print(
                f"  [{tag:<7}] {r.paper_pdf.name}  {r.model}  T={r.temperature}  "
                f"sample={r.sample}  -> {r.out}",
                flush=True,
            )
        return 0

    model_keys = args.models or list(MODELS)
    try:
        prewarm_offline_models(model_keys)
    except Exception as exc:
        print(f"ERROR: failed to load vLLM engine: {exc}", file=sys.stderr)
        return 1

    pending = [r for r in runs if not r.out.exists() or args.force]
    n_ok = n_err = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, r): r for r in pending}
        for i, fut in enumerate(as_completed(futures), 1):
            run, status, dt = fut.result()
            if status == "ok":
                n_ok += 1
            else:
                n_err += 1
            elapsed = (time.monotonic() - t0) / 60
            print(
                f"[{i:>4}/{len(pending)}] {status:<14} {dt:5.1f}s  "
                f"{run.out.name}  elapsed={elapsed:.1f}m"
            )

    total_min = (time.monotonic() - t0) / 60
    print(f"\nDone. ok={n_ok}  err={n_err}  total_minutes={total_min:.1f}")
    print(f"Timings appended to: {TIMINGS_CSV}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
