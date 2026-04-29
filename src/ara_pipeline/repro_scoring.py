"""Reproducibility index over PaperProfile JSONs.

Implements the structural / content / composite scores described in
`reproducibility_score.md`, mirroring `consistency_analysis.ipynb` ca-40..ca-42
so re-runs agree byte-for-byte.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

STEM_RE = re.compile(
    r"^(?P<paper>.+?)_T(?P<temp>[0-9p]+)_exe(?P<sample>\d+)_(?P<model>[A-Za-z0-9._-]+)$"
)
LAYER_W: dict[str, float] = {"source": 0.30, "process": 0.40, "sink": 0.30}
STRUCT_W: dict[str, float] = {
    "sources_consumed_ratio": 0.25,
    "sinks_produced_ratio": 0.25,
    "resolved_input_ratio": 0.20,
    "source_to_sink_reachability": 0.15,
    "lwcc_fraction": 0.15,
}


@dataclass(frozen=True, slots=True)
class StemInfo:
    paper: str
    temperature: float
    sample: int
    model: str


def parse_stem(stem: str) -> StemInfo | None:
    """Parse `<paper>_T<temp>_exe<sample>_<model>` (no `.profile` suffix)."""
    m = STEM_RE.match(stem)
    if m is None:
        return None
    return StemInfo(
        paper=m["paper"] + ".pdf",
        temperature=float(m["temp"].replace("p", ".")),
        sample=int(m["sample"]),
        model=m["model"].replace("_", "."),
    )


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()


def _process_id(step: dict, idx: int) -> str:
    for key in ("node_id", "step_id", "id"):
        if step.get(key):
            return str(step[key])
    return f"meth_{idx + 1:02d}"


def _proc_in(step: dict) -> list[str]:
    return list(step.get("input_ids") or step.get("data_inputs") or [])


def _proc_out(step: dict) -> list[str]:
    return list(step.get("outcomes") or step.get("expected_outputs") or [])


def build_graph(profile: dict) -> nx.DiGraph:
    g = nx.DiGraph()
    src_ids = {s["node_id"] for s in profile.get("nodes_source", [])}
    sink_ids = {s["node_id"] for s in profile.get("nodes_sink", [])}
    for sid in src_ids:
        g.add_node(sid, layer="source")
    for sid in sink_ids:
        g.add_node(sid, layer="sink")
    procs: list[tuple[str, list[str], list[str]]] = []
    for idx, step in enumerate(profile.get("nodes_process", [])):
        mid = _process_id(step, idx)
        ins = [_norm(x) for x in _proc_in(step)]
        outs = [_norm(x) for x in _proc_out(step)]
        g.add_node(mid, layer="process", input_ids=ins, outcomes=outs)
        procs.append((mid, ins, outs))
    for mid, ins, outs in procs:
        for lbl in ins:
            if lbl in src_ids:
                g.add_edge(lbl, mid)
        for lbl in outs:
            if lbl in sink_ids:
                g.add_edge(mid, lbl)
    for i, (mi, _, outs_i) in enumerate(procs):
        downstream = set(outs_i) - src_ids - sink_ids
        if not downstream:
            continue
        for j, (mj, ins_j, _) in enumerate(procs):
            if i != j and downstream & set(ins_j):
                g.add_edge(mi, mj)
    return g


def wiring_metrics(g: nx.DiGraph) -> dict[str, float]:
    by_layer: dict[str, list[str]] = {"source": [], "process": [], "sink": []}
    for v, data in g.nodes(data=True):
        by_layer[data["layer"]].append(v)
    src, proc, snk = by_layer["source"], by_layer["process"], by_layer["sink"]

    all_outs: set[str] = set()
    for v in proc:
        all_outs.update(g.nodes[v]["outcomes"])
    known = set(src) | set(snk) | all_outs

    tot_in = dangling = 0
    for v in proc:
        for lbl in g.nodes[v]["input_ids"]:
            tot_in += 1
            if lbl not in known:
                dangling += 1

    reach = (
        sum(nx.has_path(g, s, t) for s in src for t in snk) / (len(src) * len(snk))
        if src and snk
        else 0.0
    )
    n = g.number_of_nodes()
    lwcc = max((len(c) for c in nx.weakly_connected_components(g)), default=0) / max(1, n)

    src_used = sum(g.out_degree(v) > 0 for v in src) / len(src) if src else 0.0
    snk_made = sum(g.in_degree(v) > 0 for v in snk) / len(snk) if snk else 0.0
    return {
        "sources_consumed_ratio": src_used,
        "sinks_produced_ratio": snk_made,
        "resolved_input_ratio": (1 - dangling / tot_in) if tot_in else 0.0,
        "source_to_sink_reachability": reach,
        "lwcc_fraction": lwcc,
    }


def structural_score(metrics: dict[str, float]) -> float:
    return sum(STRUCT_W[k] * metrics[k] for k in STRUCT_W)


def content_score(profile: dict) -> float:
    def layer_mean(items: list[dict]) -> float | None:
        if not items:
            return None
        return sum(int(it.get("reproducibility_score", 0)) for it in items) / len(items) / 100.0

    layer = {
        "source": layer_mean(profile.get("nodes_source", [])),
        "process": layer_mean(profile.get("nodes_process", [])),
        "sink": layer_mean(profile.get("nodes_sink", [])),
    }
    present = {k: v for k, v in layer.items() if v is not None}
    if not present:
        return 0.0
    tw = sum(LAYER_W[k] for k in present)
    return sum(LAYER_W[k] * v / tw for k, v in present.items())


def score_profile(profile: dict) -> dict[str, float]:
    s = structural_score(wiring_metrics(build_graph(profile)))
    c = content_score(profile)
    return {
        "structural": s,
        "content": c,
        "repro_index": math.sqrt(max(0.0, s) * max(0.0, c)),
        "n_source": float(len(profile.get("nodes_source", []))),
        "n_process": float(len(profile.get("nodes_process", []))),
        "n_sink": float(len(profile.get("nodes_sink", []))),
    }


def load_runs(profile_dir: Path) -> list[dict]:
    """Walk `profile_dir`, score every `*.profile.json`, return one row per run."""
    rows: list[dict] = []
    for path in sorted(profile_dir.glob("*.profile.json")):
        info = parse_stem(path.stem.removesuffix(".profile"))
        if info is None:
            continue
        try:
            profile = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        rows.append(
            {
                "paper": info.paper,
                "temperature": info.temperature,
                "sample": info.sample,
                "model": info.model,
                **score_profile(profile),
            }
        )
    return rows
