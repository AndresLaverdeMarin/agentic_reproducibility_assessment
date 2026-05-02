"""Print failure rates for the consistency-check repeated runs.

Rows are models, columns are temperatures, and each value is the mean failure
rate across paper-model-temperature groups.
A group is considered complete when it has ten successful runs.
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
EXPECTED_RUNS_PER_GROUP = 10
REPORT_STD = True
REPORT_PERCENT = True


@dataclass(frozen=True)
class RunStatus:
    timestamp_utc: str
    paper: str
    model: str
    temperature: str
    sample: str
    status: str


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def read_timing_file(path: Path) -> list[RunStatus]:
    rows: list[RunStatus] = []
    if not path.is_file():
        return rows

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            paper = row.get("paper", "")
            model = row.get("model", "")
            temperature = row.get("temperature", "")
            sample = row.get("sample", "")
            if not paper or not model or temperature == "" or not sample:
                continue
            rows.append(
                RunStatus(
                    timestamp_utc=row.get("timestamp_utc", ""),
                    paper=paper,
                    model=model,
                    temperature=temperature,
                    sample=sample,
                    status=row.get("status", ""),
                )
            )
    return rows


def load_statuses() -> list[RunStatus]:
    latest_by_run: dict[tuple[str, str, str, str], RunStatus] = {}
    for filename in TIMING_FILES:
        for status in read_timing_file(INPUT_DIR / filename):
            key = (status.paper, status.model, status.temperature, status.sample)
            previous = latest_by_run.get(key)
            if previous is None or status.timestamp_utc >= previous.timestamp_utc:
                latest_by_run[key] = status
    return sorted(
        latest_by_run.values(),
        key=lambda row: (row.model, temp_sort_key(row.temperature), row.paper, row.sample),
    )


def temp_sort_key(temperature: str) -> tuple[float, str]:
    try:
        return float(temperature), temperature
    except ValueError:
        return float("inf"), temperature


def summarize_group_failures(statuses: list[RunStatus]) -> dict[tuple[str, str], list[float]]:
    by_group: dict[tuple[str, str, str], list[RunStatus]] = defaultdict(list)
    for status in statuses:
        by_group[(status.paper, status.model, status.temperature)].append(status)

    by_model_temp: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (_, model, temperature), group in sorted(by_group.items()):
        successful_runs = sum(status.status == "ok" for status in group)
        successful_runs = min(successful_runs, EXPECTED_RUNS_PER_GROUP)
        failure_rate = 1.0 - (successful_runs / EXPECTED_RUNS_PER_GROUP)
        by_model_temp[(model, temperature)].append(failure_rate)
    return by_model_temp


def summarize_models(
    group_failures: dict[tuple[str, str], list[float]],
) -> dict[str, dict[str, tuple[float, float]]]:
    table: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for (model, temperature), values in sorted(group_failures.items()):
        table[model][temperature] = (mean(values), sample_std(values))
    return dict(sorted(table.items()))


def format_failure(value: tuple[float, float] | None) -> str:
    if value is None:
        return "-"
    avg, spread = value
    scale = 100.0 if REPORT_PERCENT else 1.0
    suffix = "%" if REPORT_PERCENT else ""
    if not REPORT_STD:
        return f"{avg * scale:.1f}{suffix}"
    return f"{avg * scale:.1f}{suffix} ({spread * scale:.1f}{suffix})"


def markdown_table(table: dict[str, dict[str, tuple[float, float]]]) -> str:
    temperatures = sorted(
        {temperature for values in table.values() for temperature in values},
        key=temp_sort_key,
    )
    headers = ["model", *[f"T={temperature:g}" for temperature in map(float, temperatures)]]
    lines = [
        "## Consistency Failure Rate",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for model, values in table.items():
        row = [model]
        row.extend(format_failure(values.get(temperature)) for temperature in temperatures)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


statuses = load_statuses()
if not statuses:
    raise FileNotFoundError(f"No timing rows found in {INPUT_DIR}")

group_failures = summarize_group_failures(statuses)
model_table = summarize_models(group_failures)
print(markdown_table(model_table), end="")
