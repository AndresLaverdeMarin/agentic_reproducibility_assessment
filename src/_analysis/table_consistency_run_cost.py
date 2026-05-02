"""Print average consistency-run cost and runtime by model.

The consistency timing files record runtime for each run, but not token usage.
When a saved USD cost column is unavailable, hosted-model cost is estimated from
paper word counts and the ARA pipeline's per-token price assumptions.
Local Qwen runs are treated as zero marginal API cost.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "experiments" / "consistency_checks" / "rescience_c"
TIMING_FILES = (
    "run_timings.csv",
    "run_timings-others.csv",
    "run_timings-checkpoint.csv",
)
REPORT_STD = True
ONLY_SUCCESSFUL_RUNS = True
N_PIPELINE_CALLS = 6
WORD_TO_TOKEN_FACTOR = 1.33
PROMPT_OVERHEAD_TOKENS_PER_CALL = 1500
OUTPUT_TOKEN_RATIO = 0.20

COST_COLUMNS = (
    "total_usd",
    "cost_usd",
    "usd",
    "cost",
    "price_usd",
)

PAPER_WORD_COUNTS = {
    "2017_04_article.pdf": 3372,
    "2020_26_article.pdf": 3701,
    "2017_01_article.pdf": 6436,
    "2025_03_article.pdf": 8930,
    "2017_08_article.pdf": 9237,
    "2022_30_article.pdf": 9800,
    "2021_34_article.pdf": 13493,
    "2023_17_article.pdf": 14733,
    "2023_15_article.pdf": 16170,
    "2022_38_article.pdf": 22917,
}

PRICING_USD_PER_MTOK = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
}

MODEL_PRICE_ALIASES = {
    "gemini-3-flash-preview": "gemini-2.5-flash",
    "gemini-3.1-pro-preview": "gemini-2.5-pro",
}


@dataclass(frozen=True)
class RunTiming:
    timestamp_utc: str
    paper: str
    model: str
    temperature: str
    sample: str
    status: str
    duration_seconds: float
    cost_usd: float | None


@dataclass(frozen=True)
class PaperSummary:
    paper: str
    model: str
    n_runs: int
    runtime_seconds: float
    cost_usd: float | None


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def local_zero_cost_model(model: str) -> bool:
    return model.startswith("qwen")


def estimate_cost_usd(model: str, paper: str) -> float | None:
    if local_zero_cost_model(model):
        return 0.0

    pricing_model = MODEL_PRICE_ALIASES.get(model, model)
    rates = PRICING_USD_PER_MTOK.get(pricing_model)
    word_count = PAPER_WORD_COUNTS.get(paper)
    if rates is None or word_count is None:
        return None

    paper_tokens = word_count * WORD_TO_TOKEN_FACTOR
    input_tokens = N_PIPELINE_CALLS * (paper_tokens + PROMPT_OVERHEAD_TOKENS_PER_CALL)
    output_tokens = input_tokens * OUTPUT_TOKEN_RATIO
    input_usd = input_tokens / 1_000_000 * rates["input"]
    output_usd = output_tokens / 1_000_000 * rates["output"]
    return input_usd + output_usd


def row_cost(row: dict[str, str]) -> float | None:
    for column in COST_COLUMNS:
        value = parse_float(row.get(column))
        if value is not None:
            return value
    return estimate_cost_usd(row.get("model", ""), row.get("paper", ""))


def read_timing_file(path: Path) -> list[RunTiming]:
    rows: list[RunTiming] = []
    if not path.is_file():
        return rows

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            duration = parse_float(row.get("duration_seconds"))
            if duration is None:
                continue
            status = row.get("status", "")
            if ONLY_SUCCESSFUL_RUNS and status != "ok":
                continue
            rows.append(
                RunTiming(
                    timestamp_utc=row.get("timestamp_utc", ""),
                    paper=row.get("paper", ""),
                    model=row.get("model", ""),
                    temperature=row.get("temperature", ""),
                    sample=row.get("sample", ""),
                    status=status,
                    duration_seconds=duration,
                    cost_usd=row_cost(row),
                )
            )
    return rows


def load_timings() -> list[RunTiming]:
    latest_by_run: dict[tuple[str, str, str, str], RunTiming] = {}
    for filename in TIMING_FILES:
        for timing in read_timing_file(INPUT_DIR / filename):
            key = (timing.paper, timing.model, timing.temperature, timing.sample)
            previous = latest_by_run.get(key)
            if previous is None or timing.timestamp_utc >= previous.timestamp_utc:
                latest_by_run[key] = timing
    return sorted(
        latest_by_run.values(),
        key=lambda row: (row.model, row.paper, row.temperature, row.sample),
    )


def summarize_papers(timings: list[RunTiming]) -> list[PaperSummary]:
    grouped: dict[tuple[str, str], list[RunTiming]] = defaultdict(list)
    for timing in timings:
        grouped[(timing.model, timing.paper)].append(timing)

    summaries: list[PaperSummary] = []
    for (model, paper), rows in sorted(grouped.items()):
        known_costs = [row.cost_usd for row in rows if row.cost_usd is not None]
        summaries.append(
            PaperSummary(
                paper=paper,
                model=model,
                n_runs=len(rows),
                runtime_seconds=mean(row.duration_seconds for row in rows),
                cost_usd=mean(known_costs) if known_costs else None,
            )
        )
    return summaries


def summarize_models(papers: list[PaperSummary]) -> dict[str, dict[str, float | tuple[float, float] | None]]:
    grouped: dict[str, list[PaperSummary]] = defaultdict(list)
    for paper in papers:
        grouped[paper.model].append(paper)

    table: dict[str, dict[str, float | tuple[float, float] | None]] = {}
    for model, rows in sorted(grouped.items()):
        runtimes = [row.runtime_seconds for row in rows]
        costs = [row.cost_usd for row in rows if row.cost_usd is not None]
        table[model] = {
            "n_runs": float(sum(row.n_runs for row in rows)),
            "n_papers": float(len(rows)),
            "avg_runtime_seconds_per_paper": (mean(runtimes), sample_std(runtimes)),
            "avg_cost_usd_per_paper": (mean(costs), sample_std(costs)) if costs else None,
        }
    return table


def format_mean_std(value: float | tuple[float, float] | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if not isinstance(value, tuple):
        return f"{value:.{digits}f}"
    avg, spread = value
    if not REPORT_STD:
        return f"{avg:.{digits}f}"
    return f"{avg:.{digits}f} ({spread:.{digits}f})"


def format_cost(value: float | tuple[float, float] | None) -> str:
    if value is None:
        return "n/a"
    if not isinstance(value, tuple):
        return f"${value:.4f}"
    avg, spread = value
    if not REPORT_STD:
        return f"${avg:.4f}"
    return f"${avg:.4f} (${spread:.4f})"


def markdown_table(table: dict[str, dict[str, float | tuple[float, float] | None]]) -> str:
    headers = [
        "model",
        "n_runs",
        "n_papers",
        "avg_cost_usd_per_paper",
        "avg_runtime_seconds_per_paper",
    ]
    lines = [
        "## Consistency Run Cost and Runtime",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for model, values in table.items():
        row = [
            model,
            str(int(values["n_runs"] or 0)),
            str(int(values["n_papers"] or 0)),
            format_cost(values["avg_cost_usd_per_paper"]),
            format_mean_std(values["avg_runtime_seconds_per_paper"]),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


timings = load_timings()
if not timings:
    raise FileNotFoundError(f"No timing rows found in {INPUT_DIR}")

paper_summaries = summarize_papers(timings)
model_table = summarize_models(paper_summaries)
print(markdown_table(model_table), end="")
