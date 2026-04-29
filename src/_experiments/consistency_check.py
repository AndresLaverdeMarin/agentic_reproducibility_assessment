"""Parallel consistency experiment for the ARA paper analyst.

Runs every (model, temperature, paper, sample) combination through the analyst
and writes the resulting PaperProfile to
data/experiments/consistency_checks/rescience_c/<paper>_T<t>_exe<n>_<model>.profile.json
under the repository root. Resumes by default: existing output files are
skipped unless --force is passed. Failures land in a sidecar <stem>.err.txt so a
single bad run does not abort the sweep.

Usage (from src/):
    uv run python _experiments/consistency_check.py --workers 8
    uv run python _experiments/consistency_check.py --dry-run
    uv run python _experiments/consistency_check.py --models gemini-2.5-flash --workers 4
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
# inserting `src/` lets `from ara_pipeline import ...` resolve without an
# editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ara_pipeline import GeminiPaperAnalyst, GPTPaperAnalyst

# -- experiment grid ----------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_DIR = REPO_ROOT / "data" / "resciencec" / "papers_original"
OUTPUT_DIR = REPO_ROOT / "data" / "experiments" / "consistency_checks" / "rescience_c"
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

# Per-backend concurrency cap. The Gemini and OpenAI quotas live on different
# accounts; the global --workers value is the Gemini cap, and the GPT semaphore
# narrows OpenAI calls so a low OpenAI tier doesn't blast 429s. Reconfigured in
# main() based on --gpt-workers.
_gpt_semaphore = threading.Semaphore(2)


class _NullGate:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL_GATE = _NullGate()


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


# Selected by length to span the corpus. Trailing comments: pages, words.
PAPERS = [
    "2017_04_article.pdf",  #  4 pages,  3372 words
    "2020_26_article.pdf",  #  8 pages,  3701 words
    "2017_01_article.pdf",  # 15 pages,  6436 words
    "2025_03_article.pdf",  # 20 pages,  8930 words
    "2017_08_article.pdf",  # 11 pages,  9237 words
    "2022_30_article.pdf",  # 18 pages,  9800 words
    "2021_34_article.pdf",  # 28 pages, 13493 words
    "2023_17_article.pdf",  # 24 pages, 14733 words
    "2023_15_article.pdf",  # 30 pages, 16170 words
    "2022_38_article.pdf",  # 34 pages, 22917 words
]

# Gemini supports temperature in [0, 2]; OpenAI chat models also [0, 2];
# OSS backends typically [0, 1]; reasoning models (gpt-5*, o1/o3) ignore
# temperature -- one slot is used to drive the sample axis only.
GEMINI_TEMPS = [0.0, 0.5, 1.0, 1.5, 2.0]
GPT_TEMPS = [0.0, 0.5, 1.0, 1.5, 2.0]
GPT_REASONING_TEMPS = [0.0]
OSS_TEMPS = [0.0, 0.25, 0.5, 0.75, 1.0]

GPT_PREFIXES = ("gpt-", "o1", "o3", "o4")
GPT_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def is_gpt_model(name: str) -> bool:
    return name.startswith(GPT_PREFIXES)


def is_reasoning_model(name: str) -> bool:
    return name.startswith(GPT_REASONING_PREFIXES)


def default_temps_for(model: str) -> list[float]:
    """Fallback temperature grid for a model not pre-registered in MODELS."""
    if is_reasoning_model(model):
        return GPT_REASONING_TEMPS
    if is_gpt_model(model):
        return GPT_TEMPS
    return GEMINI_TEMPS


# Per-model temperature grid. Models passed via --models that are NOT in this
# dict still work -- they fall back to default_temps_for(model). Override here
# only when you want a non-default grid.
# OSS backends (Qwen3, Llama4, local-GPU) plug in here once their analyst
# adapters land.
MODELS: dict[str, list[float]] = {
    "gemini-2.5-flash": GEMINI_TEMPS,  # best price/perf (old generation)
    "gemini-2.5-pro": GEMINI_TEMPS,  # equivalent to 3.1 pro (old generation)
    "gemini-3-flash-preview": GEMINI_TEMPS,  # SOTA price/perf preview
    "gemini-3.1-pro-preview": GEMINI_TEMPS,  # most capable (expensive)
    "gpt-4.1-mini": GPT_TEMPS,  # OpenAI cheap chat tier
    "gpt-4.1": GPT_TEMPS,  # OpenAI flagship chat
    # "gpt-5-mini":           GPT_REASONING_TEMPS,  # reasoning model, ignores temperature
    # "qwen3-26b":            OSS_TEMPS,
    # "llama4-mm-17b":        OSS_TEMPS,
    # "local-oss-gpu":        OSS_TEMPS,
}


def make_analyst(model: str, temperature: float):
    """Dispatch to the right backend by model name."""
    if is_gpt_model(model):
        # Reasoning models (gpt-5*, o-series) reject 'temperature'.
        temp_arg = None if is_reasoning_model(model) else temperature
        return GPTPaperAnalyst(model=model, temperature=temp_arg)
    return GeminiPaperAnalyst(model=model, temperature=temperature)


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


def execute(run: Run, force: bool = False) -> tuple[Run, str, float]:
    if run.out.exists() and not force:
        append_timing(TIMINGS_CSV, run, "skip", 0.0)
        return run, "skip", 0.0
    t0 = time.monotonic()
    gate = _gpt_semaphore if is_gpt_model(run.model) else _NULL_GATE
    try:
        with gate:
            analyst = make_analyst(run.model, run.temperature)
            profile = analyst.analyze(run.paper_pdf, verbose=False)
        run.out.parent.mkdir(parents=True, exist_ok=True)
        run.out.write_text(profile.model_dump_json(indent=2))
        # Clear any prior error sidecar from a retry.
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
    selected_models = {m: MODELS.get(m, default_temps_for(m)) for m in (models or list(MODELS))}
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
        help="parallel API calls; tune to stay under your quota",
    )
    parser.add_argument(
        "--gpt-workers",
        type=int,
        default=2,
        help=(
            "max concurrent OpenAI calls (separate from --workers). "
            "Lower this if you hit RateLimitError. Tier-1 accounts often "
            "need 1-2; bump as your tier increases."
        ),
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
        help=(
            "restrict to a subset of models (default: all configured). "
            "Unregistered names are accepted and routed by prefix: 'gpt-*' / "
            "'o1*' / 'o3*' / 'o4*' -> OpenAI, anything else -> Gemini. "
            "Unregistered models use the default grid: "
            "gpt-5*/o-series -> [0.0]; other gpt-* -> [0,0.5,1,1.5,2]; "
            "rest -> [0,0.5,1,1.5,2]."
        ),
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

    global _gpt_semaphore
    _gpt_semaphore = threading.Semaphore(max(1, args.gpt_workers))

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

    n_ok = n_skip = n_err = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, r, args.force): r for r in runs}
        for i, fut in enumerate(as_completed(futures), 1):
            run, status, dt = fut.result()
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
            elapsed = (time.monotonic() - t0) / 60
            print(
                f"[{i:>4}/{len(runs)}] {status:<14} {dt:5.1f}s  "
                f"{run.out.name}  elapsed={elapsed:.1f}m"
            )

    total_min = (time.monotonic() - t0) / 60
    print(f"\nDone. ok={n_ok}  skip={n_skip}  err={n_err}  total_minutes={total_min:.1f}")
    print(f"Timings appended to: {TIMINGS_CSV}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
