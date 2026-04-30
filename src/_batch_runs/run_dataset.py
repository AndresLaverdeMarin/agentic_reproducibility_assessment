"""Run the ARA paper analyst across every PDF in a folder or zip archive.

Unlike `_experiments/consistency_check.py` (a sweep over model x temperature x
paper x sample), this script fixes a single (model, temperature) and runs the
analyst once per PDF found under --input-dir. --input-dir may be a directory
of PDFs OR a .zip archive containing PDFs (extracted to a tmp dir for the
duration of the batch and removed when the run finishes). PaperProfile JSONs
are written to --output-dir as <stem>_<temp>_<model>.profile.json and per-run
timings are appended to <output-dir>/run_timings.csv with the same columns the
consistency sweep uses (minus the 'sample' axis). Resumes by default; failures
land in a sidecar <stem>.err.txt so one bad PDF does not abort the batch.

Usage (from src/):
    uv run python _batch_runs/run_dataset.py \\
        --model gemini-2.5-flash --temperature 0.0 \\
        --input-dir ../data/resciencec/papers_original \\
        --output-dir ../data/experiments/batch_runs/rescience_c_flash_T0
    uv run python _batch_runs/run_dataset.py \\
        --model gemini-2.5-flash --temperature 0.0 \\
        --input-dir ../data/resciencec/papers_original.zip \\
        --output-dir ../data/experiments/batch_runs/rescience_c_flash_T0
    uv run python _batch_runs/run_dataset.py ... --workers 4 --force
    uv run python _batch_runs/run_dataset.py ... --retry-errors
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Make `ara_pipeline` (and `_experiments`) importable when this script is run
# directly from `src/`. `python file.py` puts the script's directory on
# sys.path, not its parent; inserting `src/` lets the imports resolve without
# an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _experiments.consistency_check import (
    is_gpt_model,
    make_analyst,
    slug_model,
    slug_temp,
)

# -- output layout ------------------------------------------------------------

TIMINGS_FIELDS = [
    "timestamp_utc",
    "paper",
    "model",
    "temperature",
    "status",
    "duration_seconds",
    "output_path",
    "error",
]
_csv_lock = threading.Lock()

# Per-backend concurrency cap mirroring the sweep script: --workers is the
# overall (Gemini) cap; the GPT semaphore narrows OpenAI calls so a low
# OpenAI tier doesn't blast 429s. Reconfigured in main() based on --gpt-workers.
_gpt_semaphore = threading.Semaphore(2)


def resolve_input(path: Path) -> tuple[Path, Path | None]:
    """Return (pdf_dir, tempdir_to_cleanup). For a .zip, extract to tmp and
    return that path as the second element so the caller can delete it after
    the batch finishes. For a directory, the second element is None."""
    if path.is_dir():
        return path, None
    if path.is_file() and path.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="ara_batch_"))
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp)
        return tmp, tmp
    raise FileNotFoundError(f"--input-dir must be a directory or a .zip file: {path}")


def output_path(out_dir: Path, paper_pdf: Path, t: float, model: str) -> Path:
    return out_dir / f"{paper_pdf.stem}_{slug_temp(t)}_{slug_model(model)}.profile.json"


# -- run unit -----------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    paper_pdf: Path
    model: str
    temperature: float
    out: Path
    timings_csv: Path


def append_timing(run: Run, status: str, duration: float, error: str = "") -> None:
    """Thread-safe append of one execution record to the timings CSV."""
    with _csv_lock:
        new_file = not run.timings_csv.exists()
        run.timings_csv.parent.mkdir(parents=True, exist_ok=True)
        with run.timings_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TIMINGS_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                    "paper": run.paper_pdf.name,
                    "model": run.model,
                    "temperature": run.temperature,
                    "status": status,
                    "duration_seconds": f"{duration:.3f}",
                    "output_path": str(run.out),
                    "error": error,
                }
            )


def execute(run: Run, force: bool = False) -> tuple[Run, str, float]:
    if run.out.exists() and not force:
        append_timing(run, "skip", 0.0)
        return run, "skip", 0.0
    err_path = run.out.with_suffix(".err.txt")
    t0 = time.monotonic()
    gate = _gpt_semaphore if is_gpt_model(run.model) else nullcontext()
    try:
        with gate:
            analyst = make_analyst(run.model, run.temperature)
            profile = analyst.analyze(run.paper_pdf, verbose=False)
        run.out.parent.mkdir(parents=True, exist_ok=True)
        run.out.write_text(profile.model_dump_json(indent=2))
        err_path.unlink(missing_ok=True)
        dt = time.monotonic() - t0
        append_timing(run, "ok", dt)
        return run, "ok", dt
    except Exception as exc:
        dt = time.monotonic() - t0
        msg = f"{type(exc).__name__}: {exc}"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(msg)
        status = f"err:{type(exc).__name__}"
        append_timing(run, status, dt, error=msg)
        return run, status, dt


def failed_stems(output_dir: Path, model: str, temperature: float) -> set[str]:
    """Paper stems that have a `.profile.err.txt` sidecar for this (model, temp)."""
    suffix = f"_{slug_temp(temperature)}_{slug_model(model)}.profile.err.txt"
    return {p.name[: -len(suffix)] for p in output_dir.glob(f"*{suffix}")}


def build_runs(
    input_dir: Path,
    output_dir: Path,
    model: str,
    temperature: float,
    pattern: str,
    only_stems: set[str] | None = None,
) -> list[Run]:
    timings_csv = output_dir / "run_timings.csv"
    pdfs = sorted(p for p in input_dir.rglob(pattern) if p.is_file())
    if only_stems is not None:
        pdfs = [p for p in pdfs if p.stem in only_stems]
    return [
        Run(
            paper_pdf=pdf,
            model=model,
            temperature=temperature,
            out=output_path(output_dir, pdf, temperature, model),
            timings_csv=timings_csv,
        )
        for pdf in pdfs
    ]


# -- main ---------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "single model id (e.g. gemini-2.5-flash, gpt-4.1-mini). "
            "Routed by prefix: 'gpt-*' / 'o1*' / 'o3*' / 'o4*' -> OpenAI, "
            "anything else -> Gemini."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        help="single temperature (ignored by reasoning models, but still recorded)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "folder of PDFs to analyze, OR a .zip file containing PDFs "
            "(extracted to a tmp dir and deleted after the batch finishes)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="folder where .profile.json files and run_timings.csv are written",
    )
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="glob pattern, applied recursively under --input-dir (default: *.pdf)",
    )
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
            "Lower this if you hit RateLimitError."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run even if the output JSON already exists",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help=(
            "only run papers that have a `.profile.err.txt` sidecar in "
            "--output-dir for this (model, temperature). Implies --force."
        ),
    )
    args = parser.parse_args()

    raw_input: Path = args.input_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    try:
        input_dir, tmp_to_clean = resolve_input(raw_input)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    global _gpt_semaphore
    _gpt_semaphore = threading.Semaphore(max(1, args.gpt_workers))

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if tmp_to_clean is not None:
            print(f"Extracted {raw_input.name} -> {input_dir}", flush=True)
        only_stems: set[str] | None = None
        force = args.force
        if args.retry_errors:
            only_stems = failed_stems(output_dir, args.model, args.temperature)
            force = True
            if not only_stems:
                print(f"  (no .err.txt sidecars found in {output_dir})", file=sys.stderr)
                return 0
            print(f"Retrying {len(only_stems)} failed paper(s) from {output_dir}", flush=True)
        runs = build_runs(
            input_dir, output_dir, args.model, args.temperature, args.pattern, only_stems
        )
        if not runs:
            print(
                f"  (no PDFs found: input_dir={input_dir} pattern={args.pattern!r}"
                f"{' retry-errors=' + str(len(only_stems)) if only_stems is not None else ''})",
                file=sys.stderr,
            )
            return 0

        n_existing = sum(1 for r in runs if r.out.exists())
        n_pending = len(runs) - n_existing
        print(
            f"Plan: {len(runs)} runs  model={args.model}  T={args.temperature}  input={raw_input}",
            flush=True,
        )
        print(
            f"      pending={n_pending}  already_done={n_existing}"
            f"{'  (use --force to re-run)' if n_existing and not force else ''}",
            flush=True,
        )

        n_ok = n_skip = n_err = 0
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(execute, r, force): r for r in runs}
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
        print(f"Timings appended to: {output_dir / 'run_timings.csv'}")
        return 0 if n_err == 0 else 1
    finally:
        if tmp_to_clean is not None:
            shutil.rmtree(tmp_to_clean, ignore_errors=True)
            print(f"Cleaned up tmp dir: {tmp_to_clean}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
