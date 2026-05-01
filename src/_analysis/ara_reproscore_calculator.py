"""Calculate ARA reproducibility scores for extracted paper-profile JSON files."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import deque
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BATCH_RUNS = ROOT / "data" / "experiments" / "batch_runs"

# Hard-coded run names. Each zip should be extracted under data/experiments/batch_runs.
RUN_NAMES = (
    "reprobench_gemini3_1_pro_T0",
    "reproscreener_gemini3_1_pro_T0",
    "rescience_c_gemini3_1_pro_T0",
)

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


def normalize_id(value: Any) -> str:
    """Normalize workflow identifiers for matching across JSON fields."""
    return re.sub(r"\s+", "", str(value or "")).lower()


def paper_id_from_path(path: Path) -> str:
    return path.name.split("_T", maxsplit=1)[0]


def node_id(node: dict[str, Any], fallback: str) -> str:
    return normalize_id(node.get("node_id") or node.get("step_id") or node.get("id") or fallback)


def input_ids(node: dict[str, Any]) -> list[str]:
    return [normalize_id(x) for x in node.get("input_ids") or node.get("data_inputs") or []]


def output_ids(node: dict[str, Any]) -> list[str]:
    return [normalize_id(x) for x in node.get("outcomes") or node.get("expected_outputs") or []]


def node_score(node: dict[str, Any]) -> float:
    """Return a node score in [0, 1]."""
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
    methods = []
    experiments = []

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


def stage_score(nodes: list[dict[str, Any]]) -> float | None:
    if not nodes:
        return None
    return sum(node_score(node) for node in nodes) / len(nodes)


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
        node_id(node, f"source_{idx}")
        for idx, node in enumerate(stages["sources"], start=1)
    }
    sink_ids = {
        node_id(node, f"sink_{idx}")
        for idx, node in enumerate(stages["sinks"], start=1)
    }

    process_nodes = stages["methods"] + stages["experiments"]
    processes = []
    for idx, process in enumerate(process_nodes, start=1):
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
            1.0 - (dangling_inputs / total_inputs)
            if total_inputs
            else 0.0
        ),
        "r_c4_source_sink_reachability": (
            reachable_pairs / source_sink_pairs
            if source_sink_pairs
            else 0.0
        ),
        "r_c5_lwcc": largest_weak_component_fraction(graph),
    }


def structural_score(components: dict[str, float]) -> float:
    return sum(STRUCTURAL_WEIGHTS[name] * components[name] for name in STRUCTURAL_WEIGHTS)


def score_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    stages = split_stages(profile)
    stage_scores = {stage: stage_score(nodes) for stage, nodes in stages.items()}

    graph_info = build_graph(stages)
    structural_parts = structural_components(graph_info)
    rc = content_score(stage_scores)
    rs = structural_score(structural_parts)
    r = math.sqrt(rc * rs)

    return {
        "paperid": paper_id_from_path(path),
        "R_sources": stage_scores["sources"],
        "R_methods": stage_scores["methods"],
        "R_experiments": stage_scores["experiments"],
        "R_sinks": stage_scores["sinks"],
        "R_c": rc,
        "R_s": rs,
        "R": r,
        **structural_parts,
        "n_sources": len(stages["sources"]),
        "n_methods": len(stages["methods"]),
        "n_experiments": len(stages["experiments"]),
        "n_sinks": len(stages["sinks"]),
        "R_0_4": 1.0 + (3.0 * r),
    }


def format_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


FIELDNAMES = [
    "paperid",
    "R_sources",
    "R_methods",
    "R_experiments",
    "R_sinks",
    "R_c",
    "R_s",
    "R",
    "r_c1_sources_consumed",
    "r_c2_sinks_produced",
    "r_c3_inputs_resolved",
    "r_c4_source_sink_reachability",
    "r_c5_lwcc",
    "n_sources",
    "n_methods",
    "n_experiments",
    "n_sinks",
    "R_0_4",
]


def json_folder_for_run(run_name: str) -> Path:
    return BATCH_RUNS / run_name / run_name


def output_csv_for_run(run_name: str) -> Path:
    return BATCH_RUNS / f"{run_name}_result.csv"


def assert_unzipped(run_name: str, folder: Path) -> None:
    if folder.exists():
        return

    zip_path = BATCH_RUNS / f"{run_name}.zip"
    if zip_path.exists():
        raise FileNotFoundError(
            f"Expected extracted JSON folder does not exist:\n"
            f"  {folder}\n"
            f"Please unzip this archive first:\n"
            f"  {zip_path}"
        )

    raise FileNotFoundError(
        f"Neither extracted JSON folder nor zip archive was found for {run_name}:\n"
        f"  {folder}\n"
        f"  {zip_path}"
    )


def write_run_scores(run_name: str) -> None:
    json_folder = json_folder_for_run(run_name)
    assert_unzipped(run_name, json_folder)

    rows = [score_profile(path) for path in sorted(json_folder.glob("*.profile.json"))]
    if not rows:
        raise FileNotFoundError(f"No *.profile.json files found in {json_folder}")

    out_csv = output_csv_for_run(run_name)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_cell(row[key]) for key in FIELDNAMES})

    print(f"Wrote {len(rows)} rows to {out_csv}")


def main() -> None:
    for run_name in RUN_NAMES:
        write_run_scores(run_name)


if __name__ == "__main__":
    main()
