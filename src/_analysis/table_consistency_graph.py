"""Summarise workflow-graph consistency across ARA repeated runs.

This script combines all available papers into one model-level table averaged
across temperatures. Each paper-model-temperature group is first reduced to
within-group consistency statistics, then those group-level values are averaged
per model. This keeps papers with more successful executions from dominating the
summary.

Run from the repository root or from ``src/``:

    python src/_analysis/table_consistency_graph.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "data" / "experiments" / "consistency_checks" / "rescience_c"
PAPER_ID: str | None = None
REPORT_STD = False
TEMPERATURE_ZERO = "T0"

COUNT_METRICS = (
    "n_nodes_total",
    "n_nodes_sources",
    "n_nodes_methods",
    "n_nodes_experiments",
    "n_nodes_sinks",
    "n_edges",
)

SUMMARY_METRICS = (
    "avg_normalized_ged",
    "avg_n_edges_variability",
    "avg_n_nodes_total_variability",
    "avg_n_nodes_sources_variability",
    "avg_n_nodes_methods_variability",
    "avg_n_nodes_experiments_variability",
    "avg_n_nodes_sinks_variability",
)

FILENAME_RE = re.compile(
    r"^(?P<paper>.+)_(?P<temperature>T[^_]+)_exe(?P<sample>\d+)_(?P<model>.+)\.profile\.json$"
)


@dataclass(frozen=True)
class RunInfo:
    path: Path
    paper: str
    temperature: str
    sample: int
    model: str


@dataclass(frozen=True)
class GraphMetrics:
    run: RunInfo
    n_nodes_total: int
    n_nodes_sources: int
    n_nodes_methods: int
    n_nodes_experiments: int
    n_nodes_sinks: int
    n_edges: int
    nodes: frozenset[str]
    edges: frozenset[tuple[str, str]]

    def counts(self) -> dict[str, float]:
        return {
            "n_nodes_total": float(self.n_nodes_total),
            "n_nodes_sources": float(self.n_nodes_sources),
            "n_nodes_methods": float(self.n_nodes_methods),
            "n_nodes_experiments": float(self.n_nodes_experiments),
            "n_nodes_sinks": float(self.n_nodes_sinks),
            "n_edges": float(self.n_edges),
        }

    @property
    def graph_size(self) -> int:
        return len(self.nodes) + len(self.edges)


@dataclass(frozen=True)
class GroupSummary:
    paper: str
    temperature: str
    model: str
    n_runs: int
    normalized_ged: float
    n_nodes_total_variability: float
    n_nodes_sources_variability: float
    n_nodes_methods_variability: float
    n_nodes_experiments_variability: float
    n_nodes_sinks_variability: float
    n_edges_variability: float

    def values(self) -> dict[str, float]:
        return {
            "avg_normalized_ged": self.normalized_ged,
            "avg_n_nodes_total_variability": self.n_nodes_total_variability,
            "avg_n_nodes_sources_variability": self.n_nodes_sources_variability,
            "avg_n_nodes_methods_variability": self.n_nodes_methods_variability,
            "avg_n_nodes_experiments_variability": self.n_nodes_experiments_variability,
            "avg_n_nodes_sinks_variability": self.n_nodes_sinks_variability,
            "avg_n_edges_variability": self.n_edges_variability,
        }


def normalize_id(value: Any) -> str:
    """Normalize workflow identifiers for matching across JSON fields."""
    return re.sub(r"\s+", "", str(value or "")).lower()


def display_model(slug: str) -> str:
    """Undo the filename slug used for dots in model names."""
    return slug.replace("_", ".")


def display_temperature(slug: str) -> str:
    return slug[1:].replace("p", ".") if slug.startswith("T") else slug


def parse_run(path: Path) -> RunInfo | None:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    return RunInfo(
        path=path,
        paper=match.group("paper"),
        temperature=match.group("temperature"),
        sample=int(match.group("sample")),
        model=match.group("model"),
    )


def node_id(node: dict[str, Any], fallback: str) -> str:
    return normalize_id(node.get("node_id") or node.get("step_id") or node.get("id") or fallback)


def input_ids(node: dict[str, Any]) -> list[str]:
    return [normalize_id(x) for x in node.get("input_ids") or node.get("data_inputs") or []]


def output_ids(node: dict[str, Any]) -> list[str]:
    return [normalize_id(x) for x in node.get("outcomes") or node.get("expected_outputs") or []]


def split_stages(profile: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    sources = list(profile.get("nodes_source", []))
    sinks = list(profile.get("nodes_sink", []))
    methods: list[dict[str, Any]] = []
    experiments: list[dict[str, Any]] = []

    for process in profile.get("nodes_process", []):
        process_type = str(process.get("process_type") or "").lower()
        if process_type == "experiment":
            experiments.append(process)
        else:
            methods.append(process)

    return {
        "sources": sources,
        "methods": methods,
        "experiments": experiments,
        "sinks": sinks,
    }


def add_edge(edges: set[tuple[str, str]], nodes: set[str], source: str, target: str) -> None:
    nodes.add(source)
    nodes.add(target)
    edges.add((source, target))


def build_graph(stages: dict[str, list[dict[str, Any]]]) -> tuple[set[str], set[tuple[str, str]]]:
    source_ids = {
        node_id(node, f"source_{idx}") for idx, node in enumerate(stages["sources"], start=1)
    }
    sink_ids = {
        node_id(node, f"sink_{idx}") for idx, node in enumerate(stages["sinks"], start=1)
    }

    processes: list[tuple[str, list[str], list[str]]] = []
    for idx, process in enumerate(stages["methods"] + stages["experiments"], start=1):
        processes.append((node_id(process, f"process_{idx}"), input_ids(process), output_ids(process)))

    nodes = set(source_ids) | set(sink_ids) | {pid for pid, _, _ in processes}
    edges: set[tuple[str, str]] = set()

    for pid, inputs, outputs in processes:
        for input_id in inputs:
            if input_id in source_ids:
                add_edge(edges, nodes, input_id, pid)
        for output_id in outputs:
            if output_id in sink_ids:
                add_edge(edges, nodes, pid, output_id)

    for source_pid, _, outputs in processes:
        output_set = set(outputs) - source_ids - sink_ids
        if not output_set:
            continue
        for target_pid, inputs, _ in processes:
            if source_pid != target_pid and output_set & set(inputs):
                add_edge(edges, nodes, source_pid, target_pid)

    return nodes, edges


def graph_edit_distance(left: GraphMetrics, right: GraphMetrics) -> int:
    """Labeled graph edit distance using insert/delete cost 1 for nodes and edges."""
    return len(left.nodes ^ right.nodes) + len(left.edges ^ right.edges)


def load_run_metrics(run: RunInfo) -> GraphMetrics:
    profile = json.loads(run.path.read_text(encoding="utf-8"))
    stages = split_stages(profile)
    nodes, edges = build_graph(stages)

    return GraphMetrics(
        run=run,
        n_nodes_total=(
            len(stages["sources"])
            + len(stages["methods"])
            + len(stages["experiments"])
            + len(stages["sinks"])
        ),
        n_nodes_sources=len(stages["sources"]),
        n_nodes_methods=len(stages["methods"]),
        n_nodes_experiments=len(stages["experiments"]),
        n_nodes_sinks=len(stages["sinks"]),
        n_edges=len(edges),
        nodes=frozenset(nodes),
        edges=frozenset(edges),
    )


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def summarize_group(group: list[GraphMetrics]) -> GroupSummary:
    first = group[0]
    pairwise_distances = [graph_edit_distance(left, right) for left, right in combinations(group, 2)]
    avg_ged = mean(pairwise_distances) if pairwise_distances else 0.0
    avg_graph_size = mean([metric.graph_size for metric in group])
    normalized_ged = avg_ged / avg_graph_size if avg_graph_size else 0.0
    count_variability = {
        metric: sample_std([row.counts()[metric] for row in group])
        for metric in COUNT_METRICS
    }

    return GroupSummary(
        paper=first.run.paper,
        temperature=first.run.temperature,
        model=first.run.model,
        n_runs=len(group),
        normalized_ged=normalized_ged,
        n_nodes_total_variability=count_variability["n_nodes_total"],
        n_nodes_sources_variability=count_variability["n_nodes_sources"],
        n_nodes_methods_variability=count_variability["n_nodes_methods"],
        n_nodes_experiments_variability=count_variability["n_nodes_experiments"],
        n_nodes_sinks_variability=count_variability["n_nodes_sinks"],
        n_edges_variability=count_variability["n_edges"],
    )


def summarize_groups(metrics: list[GraphMetrics]) -> list[GroupSummary]:
    grouped: dict[tuple[str, str, str], list[GraphMetrics]] = defaultdict(list)
    for metric in metrics:
        grouped[(metric.run.paper, metric.run.model, metric.run.temperature)].append(metric)

    return [
        summarize_group(sorted(group, key=lambda row: row.run.sample))
        for _, group in sorted(grouped.items())
    ]


def summarize_models(groups: list[GroupSummary]) -> dict[str, dict[str, float | tuple[float, float]]]:
    grouped: dict[str, list[GroupSummary]] = defaultdict(list)
    for group in groups:
        grouped[group.model].append(group)

    table: dict[str, dict[str, float | tuple[float, float]]] = {}
    for model, model_groups in sorted(grouped.items()):
        table[model] = {
            "n_runs": float(sum(group.n_runs for group in model_groups)),
            "n_groups": float(len(model_groups)),
        }
        for metric in SUMMARY_METRICS:
            values = [group.values()[metric] for group in model_groups]
            table[model][metric] = (mean(values), sample_std(values))
    return table


def format_mean_std(value: float | tuple[float, float]) -> str:
    if not isinstance(value, tuple):
        return f"{value:.2f}"
    avg, spread = value
    if not REPORT_STD:
        return f"{avg:.2f}"
    return f"{avg:.2f} ({spread:.2f})"


def markdown_table(table: dict[str, dict[str, float | tuple[float, float]]], title: str) -> str:
    headers = ["model", "n_runs", *SUMMARY_METRICS]
    lines = [
        f"## {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for model, values in table.items():
        row = [display_model(model), str(int(values["n_runs"]))]
        row.extend(format_mean_std(values[metric]) for metric in SUMMARY_METRICS)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def collect_runs(input_dir: Path, paper_id: str | None) -> list[RunInfo]:
    runs = []
    for path in sorted(input_dir.glob("*.profile.json")):
        run = parse_run(path)
        if run is not None and (paper_id is None or run.paper == paper_id):
            runs.append(run)
    return runs


runs = collect_runs(DEFAULT_INPUT_DIR, PAPER_ID)
if not runs:
    scope = f"paper '{PAPER_ID}'" if PAPER_ID else "any paper"
    raise FileNotFoundError(f"No profile JSON files found for {scope} in {DEFAULT_INPUT_DIR}")

metrics = [load_run_metrics(run) for run in runs]
groups = summarize_groups(metrics)
all_temperature_table = summarize_models(groups)
temperature_zero_groups = [group for group in groups if group.temperature == TEMPERATURE_ZERO]
temperature_zero_table = summarize_models(temperature_zero_groups)
markdown = "\n".join(
    [
        markdown_table(
            all_temperature_table,
            "Graph Consistency Averaged Across Papers and Temperatures",
        ).rstrip(),
        "",
        markdown_table(
            temperature_zero_table,
            "Graph Consistency Averaged Across Papers at T=0",
        ).rstrip(),
        "",
    ]
)

print(markdown, end="")
