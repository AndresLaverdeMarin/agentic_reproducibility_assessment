"""Print reproducibility-score consistency across repeated ARA runs.

The script prints two Markdown tables to the console only.
The first table averages across all temperatures, and the second table uses
temperature zero only.
Each paper-model-temperature group is first reduced to the sample standard
deviation of each score across repeated runs, then model-level means are
computed across groups.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "experiments" / "consistency_checks" / "rescience_c"
PAPER_ID: str | None = None
TEMPERATURE_ZERO = "T0"
REPORT_STD = False

STAGE_WEIGHTS = {
    "sources": 0.30,
    "methods": 0.20,
    "experiments": 0.20,
    "sinks": 0.30,
}

STRUCTURAL_WEIGHTS = {
    "r_c1_sources_consumed": 0.25,
    "r_c2_sinks_produced": 0.25,
    "r_c3_inputs_resolved": 0.20,
    "r_c4_source_sink_reachability": 0.15,
    "r_c5_lwcc": 0.15,
}

RUN_SCORE_METRICS = (
    "avg_node_score_all",
    "avg_node_score_sources",
    "avg_node_score_methods",
    "avg_node_score_experiments",
    "avg_node_score_sinks",
    "R_c",
    "R_s",
    "R",
)

TABLE_METRICS = (
    "std_node_score_all",
    "std_node_score_sources",
    "std_node_score_methods",
    "std_node_score_experiments",
    "std_node_score_sinks",
    "std_R_c",
    "std_R_s",
    "std_R",
)

RUN_TO_TABLE_METRIC = dict(zip(RUN_SCORE_METRICS, TABLE_METRICS, strict=True))

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
class RunScores:
    run: RunInfo
    values: dict[str, float | None]


@dataclass(frozen=True)
class GroupScores:
    paper: str
    temperature: str
    model: str
    n_runs: int
    values: dict[str, float | None]


def normalize_id(value: Any) -> str:
    """Normalize workflow identifiers for matching across JSON fields."""
    return re.sub(r"\s+", "", str(value or "")).lower()


def display_model(slug: str) -> str:
    return slug.replace("_", ".")


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


def node_score(node: dict[str, Any]) -> float:
    """Return the normalized node-level reproducibility score in [0, 1]."""
    if "reproducibility_score" in node and node["reproducibility_score"] is not None:
        return float(node["reproducibility_score"]) / 100.0
    if "algorithm_clarity" in node and node["algorithm_clarity"] is not None:
        return float(node["algorithm_clarity"]) / 4.0
    if "statement_clarity" in node and node["statement_clarity"] is not None:
        return float(node["statement_clarity"]) / 4.0
    return 0.0


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


def score_nodes(nodes: list[dict[str, Any]]) -> float | None:
    if not nodes:
        return None
    return mean(node_score(node) for node in nodes)


def content_score(stage_scores: dict[str, float | None]) -> float:
    present_stages = [stage for stage, score in stage_scores.items() if score is not None]
    if not present_stages:
        return 0.0

    present_weight = sum(STAGE_WEIGHTS[stage] for stage in present_stages)
    return sum(
        (STAGE_WEIGHTS[stage] / present_weight) * float(stage_scores[stage])
        for stage in present_stages
    )


def add_edge(graph: dict[str, set[str]], source: str, target: str) -> None:
    graph.setdefault(source, set()).add(target)
    graph.setdefault(target, set())


def build_graph(stages: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    source_ids = {
        node_id(node, f"source_{idx}") for idx, node in enumerate(stages["sources"], start=1)
    }
    sink_ids = {
        node_id(node, f"sink_{idx}") for idx, node in enumerate(stages["sinks"], start=1)
    }

    processes = []
    for idx, process in enumerate(stages["methods"] + stages["experiments"], start=1):
        pid = node_id(process, f"process_{idx}")
        processes.append((pid, input_ids(process), output_ids(process)))

    graph = {node: set() for node in source_ids | sink_ids | {pid for pid, _, _ in processes}}

    for pid, inputs, outputs in processes:
        for input_id in inputs:
            if input_id in source_ids:
                add_edge(graph, input_id, pid)
        for output_id in outputs:
            if output_id in sink_ids:
                add_edge(graph, pid, output_id)

    for source_pid, _, outputs in processes:
        output_set = set(outputs) - source_ids - sink_ids
        if not output_set:
            continue
        for target_pid, inputs, _ in processes:
            if source_pid != target_pid and output_set & set(inputs):
                add_edge(graph, source_pid, target_pid)

    all_outputs = {output for _, _, outputs in processes for output in outputs}
    known_input_ids = source_ids | sink_ids | all_outputs

    return {
        "graph": graph,
        "source_ids": source_ids,
        "sink_ids": sink_ids,
        "processes": processes,
        "known_input_ids": known_input_ids,
    }


def has_path(graph: dict[str, set[str]], source: str, target: str) -> bool:
    seen = set()
    queue = deque([source])
    while queue:
        current = queue.popleft()
        if current == target:
            return True
        if current in seen:
            continue
        seen.add(current)
        queue.extend(graph.get(current, set()) - seen)
    return False


def largest_weak_component_fraction(graph: dict[str, set[str]]) -> float:
    if not graph:
        return 0.0

    undirected = {node: set(neighbors) for node, neighbors in graph.items()}
    for source, targets in graph.items():
        for target in targets:
            undirected.setdefault(target, set()).add(source)
            undirected.setdefault(source, set()).add(target)

    seen = set()
    largest = 0
    for node in undirected:
        if node in seen:
            continue
        size = 0
        queue = deque([node])
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            size += 1
            queue.extend(undirected[current] - seen)
        largest = max(largest, size)

    return largest / len(undirected)


def structural_components(graph_info: dict[str, Any]) -> dict[str, float]:
    graph = graph_info["graph"]
    source_ids = graph_info["source_ids"]
    sink_ids = graph_info["sink_ids"]
    processes = graph_info["processes"]
    known_input_ids = graph_info["known_input_ids"]

    in_degree = {node: 0 for node in graph}
    for targets in graph.values():
        for target in targets:
            in_degree[target] = in_degree.get(target, 0) + 1

    total_inputs = sum(len(inputs) for _, inputs, _ in processes)
    dangling_inputs = sum(
        input_id not in known_input_ids
        for _, inputs, _ in processes
        for input_id in inputs
    )
    source_sink_pairs = len(source_ids) * len(sink_ids)
    reachable_pairs = sum(
        has_path(graph, source_id, sink_id)
        for source_id in source_ids
        for sink_id in sink_ids
    )

    return {
        "r_c1_sources_consumed": (
            sum(len(graph.get(source_id, set())) > 0 for source_id in source_ids) / len(source_ids)
            if source_ids
            else 0.0
        ),
        "r_c2_sinks_produced": (
            sum(in_degree.get(sink_id, 0) > 0 for sink_id in sink_ids) / len(sink_ids)
            if sink_ids
            else 0.0
        ),
        "r_c3_inputs_resolved": (
            1.0 - (dangling_inputs / total_inputs) if total_inputs else 0.0
        ),
        "r_c4_source_sink_reachability": (
            reachable_pairs / source_sink_pairs if source_sink_pairs else 0.0
        ),
        "r_c5_lwcc": largest_weak_component_fraction(graph),
    }


def structural_score(components: dict[str, float]) -> float:
    return sum(STRUCTURAL_WEIGHTS[name] * components[name] for name in STRUCTURAL_WEIGHTS)


def score_run(run: RunInfo) -> RunScores:
    profile = json.loads(run.path.read_text(encoding="utf-8"))
    stages = split_stages(profile)
    stage_scores = {stage: score_nodes(nodes) for stage, nodes in stages.items()}
    all_nodes = stages["sources"] + stages["methods"] + stages["experiments"] + stages["sinks"]
    rc = content_score(stage_scores)
    rs = structural_score(structural_components(build_graph(stages)))
    r = math.sqrt(rc * rs)

    return RunScores(
        run=run,
        values={
            "avg_node_score_all": score_nodes(all_nodes),
            "avg_node_score_sources": stage_scores["sources"],
            "avg_node_score_methods": stage_scores["methods"],
            "avg_node_score_experiments": stage_scores["experiments"],
            "avg_node_score_sinks": stage_scores["sinks"],
            "R_c": rc,
            "R_s": rs,
            "R": r,
        },
    )


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def summarize_groups(scores: list[RunScores]) -> list[GroupScores]:
    grouped: dict[tuple[str, str, str], list[RunScores]] = defaultdict(list)
    for score in scores:
        grouped[(score.run.paper, score.run.model, score.run.temperature)].append(score)

    summaries = []
    for (_, _, _), group in sorted(grouped.items()):
        first = group[0].run
        values: dict[str, float | None] = {}
        for run_metric, table_metric in RUN_TO_TABLE_METRIC.items():
            metric_values = [
                score.values[run_metric]
                for score in group
                if score.values[run_metric] is not None
            ]
            values[table_metric] = sample_std(metric_values) if metric_values else None
        summaries.append(
            GroupScores(
                paper=first.paper,
                temperature=first.temperature,
                model=first.model,
                n_runs=len(group),
                values=values,
            )
        )
    return summaries


def summarize_models(groups: list[GroupScores]) -> dict[str, dict[str, float | tuple[float, float] | None]]:
    grouped: dict[str, list[GroupScores]] = defaultdict(list)
    for group in groups:
        grouped[group.model].append(group)

    table: dict[str, dict[str, float | tuple[float, float] | None]] = {}
    for model, model_groups in sorted(grouped.items()):
        table[model] = {"n_runs": float(sum(group.n_runs for group in model_groups))}
        for metric in TABLE_METRICS:
            values = [group.values[metric] for group in model_groups if group.values[metric] is not None]
            table[model][metric] = (mean(values), sample_std(values)) if values else None
    return table


def format_mean_std(value: float | tuple[float, float] | None) -> str:
    if value is None:
        return "-"
    if not isinstance(value, tuple):
        return f"{value:.2f}"
    avg, spread = value
    if not REPORT_STD:
        return f"{avg:.2f}"
    return f"{avg:.2f} ({spread:.2f})"


def markdown_table(
    table: dict[str, dict[str, float | tuple[float, float] | None]],
    title: str,
) -> str:
    headers = ["model", "n_runs", *TABLE_METRICS]
    lines = [
        f"## {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for model, values in table.items():
        row = [display_model(model), str(int(values["n_runs"] or 0))]
        row.extend(format_mean_std(values[metric]) for metric in TABLE_METRICS)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def collect_runs(input_dir: Path, paper_id: str | None) -> list[RunInfo]:
    runs = []
    for path in sorted(input_dir.glob("*.profile.json")):
        run = parse_run(path)
        if run is not None and (paper_id is None or run.paper == paper_id):
            runs.append(run)
    return runs


runs = collect_runs(INPUT_DIR, PAPER_ID)
if not runs:
    scope = f"paper '{PAPER_ID}'" if PAPER_ID else "any paper"
    raise FileNotFoundError(f"No profile JSON files found for {scope} in {INPUT_DIR}")

run_scores = [score_run(run) for run in runs]
groups = summarize_groups(run_scores)
all_temperature_table = summarize_models(groups)
temperature_zero_table = summarize_models(
    [group for group in groups if group.temperature == TEMPERATURE_ZERO]
)

markdown = "\n".join(
    [
        markdown_table(
            all_temperature_table,
            "Reproducibility Score Variability Averaged Across Papers and Temperatures",
        ).rstrip(),
        "",
        markdown_table(
            temperature_zero_table,
            "Reproducibility Score Variability Averaged Across Papers at T=0",
        ).rstrip(),
        "",
    ]
)
print(markdown, end="")
