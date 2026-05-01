"""Run the customized ARA pipeline over the ReproScreener PDF folder.

Spyder-friendly usage:
1. Put a ``.env`` file next to this file with ``GEMINI_API_KEY``/``GOOGLE_API_KEY``
   for Gemini.
2. Adjust the configuration constants below if desired.
3. Run this file.

Outputs are written to ``OUTPUT_DIR`` as ``<paper-stem>_<temp>_<model>.profile.json``.
Existing successful outputs are skipped unless ``FORCE_RERUN`` is set to True.
"""

from __future__ import annotations

import csv
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if load_dotenv is not None:
    load_dotenv(THIS_DIR / ".env")
    load_dotenv(REPO_ROOT / ".env")

from ara_customized_pipeline import make_customized_analyst


# ---- Configuration ---------------------------------------------------------

MODEL = "gemini-3.1-pro-preview"
TEMPERATURE: float | None = 0.0
FORCE_RERUN = False
VERBOSE_PER_PAPER = False
WORKERS = 10

INPUT_DIR = REPO_ROOT / "data" / "reproscreener" / "reproscreener_papers_original" / "papers_original"
OUTPUT_DIR = THIS_DIR / "outputs"
TIMINGS_CSV = OUTPUT_DIR / "run_timings.csv"
_csv_lock = threading.Lock()


def slug_temp(t: float | None) -> str:
    if t is None:
        return "Tnone"
    return f"T{str(float(t)).replace('.', 'p')}"


def slug_model(model: str) -> str:
    return model.replace(".", "_").replace("-", "_")


def output_path(pdf_path: Path) -> Path:
    return OUTPUT_DIR / f"{pdf_path.stem}_{slug_temp(TEMPERATURE)}_{slug_model(MODEL)}.profile.json"


def append_timing(
    paper: Path,
    status: str,
    duration_seconds: float,
    out_path: Path,
    error: str = "",
) -> None:
    fieldnames = [
        "timestamp_utc",
        "paper",
        "model",
        "temperature",
        "status",
        "duration_seconds",
        "output_path",
        "error",
    ]
    with _csv_lock:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        new_file = not TIMINGS_CSV.exists()
        with TIMINGS_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if new_file:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                    "paper": paper.name,
                    "model": MODEL,
                    "temperature": TEMPERATURE,
                    "status": status,
                    "duration_seconds": f"{duration_seconds:.3f}",
                    "output_path": str(out_path),
                    "error": error,
                }
            )


def run_one(pdf_path: Path) -> str:
    out_path = output_path(pdf_path)
    err_path = out_path.with_suffix(".err.txt")
    if out_path.exists() and not FORCE_RERUN:
        append_timing(pdf_path, "skip", 0.0, out_path)
        print(f"SKIP {pdf_path.name} -> {out_path.name}")
        return "skip"

    started = time.monotonic()
    try:
        analyst = make_customized_analyst(model=MODEL, temperature=TEMPERATURE)
        profile = analyst.analyze(pdf_path, verbose=VERBOSE_PER_PAPER)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
        err_path.unlink(missing_ok=True)
        duration = time.monotonic() - started
        append_timing(pdf_path, "ok", duration, out_path)
        print(f"OK   {pdf_path.name} -> {out_path.name} ({duration:.1f}s)")
        return "ok"
    except Exception as exc:
        duration = time.monotonic() - started
        msg = f"{type(exc).__name__}: {exc}"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        err_path.write_text(msg, encoding="utf-8")
        append_timing(pdf_path, f"err:{type(exc).__name__}", duration, out_path, error=msg)
        print(f"ERR  {pdf_path.name}: {msg}")
        return "err"


def main() -> int:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Input PDF folder not found: {INPUT_DIR}")

    pdfs = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {INPUT_DIR}")
        return 0

    print(f"Customized ARA ReproScreener run")
    print(f"Backend: Gemini  model={MODEL}  temperature={TEMPERATURE}")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Papers: {len(pdfs)}")
    print(f"Workers: {WORKERS}  force_rerun={FORCE_RERUN}")

    counts = {"ok": 0, "skip": 0, "err": 0}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, int(WORKERS))) as pool:
        futures = {pool.submit(run_one, pdf_path): pdf_path for pdf_path in pdfs}
        for index, future in enumerate(as_completed(futures), start=1):
            pdf_path = futures[future]
            try:
                status = future.result()
            except Exception as exc:
                status = "err"
                print(f"ERR  {pdf_path.name}: {type(exc).__name__}: {exc}")
            counts[status] += 1
            elapsed_minutes = (time.monotonic() - started) / 60
            print(
                f"[{index}/{len(pdfs)}] {status.upper():<4} {pdf_path.name} "
                f"elapsed={elapsed_minutes:.1f}m"
            )

    minutes = (time.monotonic() - started) / 60
    print(
        f"\nDone. ok={counts['ok']} skip={counts['skip']} "
        f"err={counts['err']} total_minutes={minutes:.1f}"
    )
    print(f"Timings: {TIMINGS_CSV}")
    return 0 if counts["err"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
