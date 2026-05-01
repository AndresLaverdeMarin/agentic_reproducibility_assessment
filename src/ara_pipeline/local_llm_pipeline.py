"""Local LLM pipeline for reproducibility analysis of a research paper.

Runs offline vLLM inference (no HTTP server required) using Qwen3 models.
Guided JSON decoding enforces Pydantic schema conformance natively.
"""

from __future__ import annotations

import threading
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class DataAvailability(StrEnum):
    OPEN = "open"
    UPON_REQUEST = "upon_request"
    NOT_MENTIONED = "not_mentioned"


class SinkType(StrEnum):
    FIGURE = "figure"
    TABLE = "table"


class ProcessType(StrEnum):
    METHOD = "method"
    EXPERIMENT = "experiment"


class StatementValidity(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    NOT_ASSESSED = "not_assessed"


class ProcessNode(BaseModel):
    node_id: str = Field(
        pattern=r"^meth_\d{2}$",
        description="Canonical step id 'meth_<NN>' (zero-padded, two digits). "
        "First step is 'meth_01'; increment strictly in sequential execution order.",
    )
    node_name: str = Field(description="Short human-readable name (<=8 words).")
    source_quote: str = Field(description="Short literal quote grounding this step.")
    description: str
    process_type: ProcessType = Field(
        default=ProcessType.METHOD,
        description="Classification of the step: 'method' for an algorithmic / "
        "computational procedure (training, inference, preprocessing, feature "
        "extraction, optimisation), or 'experiment' for a controlled trial / "
        "evaluation / ablation / comparison / hyperparameter sweep that "
        "produces measured outcomes.",
    )
    input_ids: list[str] = Field(
        default_factory=list,
        description="Labels consumed by the step: a source id ('src_*'), a sink id "
        "('sink_*'), or the verbatim label emitted as an 'outcomes' item by an earlier step.",
    )
    outcomes: list[str] = Field(
        default_factory=list,
        description="Labels produced by the step. Intermediates are lowercase snake_case "
        "names that downstream steps reuse verbatim in 'input_ids'; the final step of a "
        "chain emits a sink_* id.",
    )
    algorithm_clarity: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Ordinal reconstructability score r(.) in {1,2,3,4}. "
        "1=missing, 2=partial, 3=mostly specified, 4=sufficient for independent reconstruction.",
    )
    tools_required: list[str] = Field(
        default_factory=list,
        description="Generic technique(s) needed to execute the step (e.g. 'gradient-"
        "boosted classifier', 'stratified k-fold CV'), independent of any specific library.",
    )
    tools_mentioned: list[str] = Field(
        default_factory=list,
        description="Concrete tool / library / framework names the paper actually names "
        "for this step, with versions if stated (else 'version not stated').",
    )
    parameters_required: list[str] = Field(
        default_factory=list,
        description="Names of parameters the technique needs to be reproducible "
        "(e.g. 'learning_rate', 'random_seed'), regardless of whether the paper gives values.",
    )
    parameters_mentioned: list[str] = Field(
        default_factory=list,
        description="Parameters the paper states for this step, as 'name=value' strings.",
    )
    reproducibility_score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Percentage in [0, 100] estimating how reproducibly the step is "
        "specified, integrating 'algorithm_clarity', the gap between 'tools_required' "
        "and 'tools_mentioned', and the gap between 'parameters_required' and "
        "'parameters_mentioned'. 0 = none of these are usable; 100 = clarity is 4, "
        "every required tool is named, and every required parameter has a value.",
    )
    reproducibility_rationale: str = Field(
        default="",
        description="Short justification (<=200 chars) for the assigned "
        "'reproducibility_score', citing the specific gaps that drove the value "
        "(e.g. 'clarity=3, 1/2 tools named, all params stated').",
    )


class Dataset(BaseModel):
    node_id: str = Field(
        pattern=r"^src_[a-z0-9_]{1,40}$",
        description="Canonical slug, lowercase ASCII, prefix 'src_', derived from the most "
        "specific noun phrase naming this dataset in the paper. Used verbatim by "
        "nodes_process[*].input_ids to wire sources into the workflow graph.",
    )
    node_name: str = Field(description="Human-readable dataset name as printed in the paper.")
    source_quote: str = Field(description="Literal quote mentioning this dataset.")
    description: str = ""
    size_estimate: str | None = None
    license: str | None = None
    availability: DataAvailability = DataAvailability.NOT_MENTIONED
    url: str | None = None
    reproducibility_score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Percentage in [0, 100] estimating how reproducibly accessible this "
        "dataset is. 0 = not available / not mentioned (no source, no URL, no "
        "acquisition path stated). 100 = openly mentioned with a working source / "
        "URL / DOI a third party can use to obtain the exact dataset. "
        "Intermediate values reflect partial information (e.g. named source but no "
        "URL, upon-request access, citation-only).",
    )
    reproducibility_rationale: str = Field(
        default="",
        description="Short justification (<=200 chars) for the assigned "
        "'reproducibility_score', citing the specific facts that drove the value "
        "(e.g. 'open URL provided', 'cited reference only, no link', "
        "'upon-request, no DOI').",
    )


class SinkNode(BaseModel):
    """Reproducible artefact (figure or table) the paper reports."""

    node_id: str = Field(
        pattern=r"^sink_(fig|tab)[a-z0-9_]*$",
        description="Canonical slug: 'sink_fig<N>' for figures, 'sink_tab<N>' for tables. "
        "Used verbatim by nodes_process[*].outcomes to wire the producing step into the graph.",
    )
    node_name: str = Field(
        description="Paper's printed label for the artefact, e.g. 'Figure 3', 'Table 1'.",
    )
    source_quote: str = Field(description="Short literal quote anchoring the artefact in the PDF.")
    description: str = Field(default="", description="What the figure/table shows or presents.")
    input_ids: list[str] = Field(
        default_factory=list,
        description="Predecessor labels the artefact is built from: source ids ('src_*') used "
        "verbatim, and/or intermediate artefact labels matching a method step's 'outcomes'.",
    )
    size_estimate: str | None = Field(
        default=None,
        description="Approximate size if stated (e.g. 'N=120 cells x 6 columns', '8 panels').",
    )
    type: SinkType = Field(description="'figure' or 'table'.")
    statement_clarity: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Ordinal reconstructability score r(.) in {1,2,3,4} for the procedure "
        "FROM 'input_ids' TO this artefact. "
        "1=missing, 2=partial, 3=mostly specified, 4=sufficient for independent reconstruction.",
    )
    statement_validity: StatementValidity = Field(
        default=StatementValidity.NOT_ASSESSED,
        description="Whether the artefact's conclusions / data are validly supported "
        "by 'input_ids'.",
    )
    reproducibility_score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Percentage in [0, 100] estimating how reproducibly this result "
        "could be regenerated. Combines (a) the upstream reproducibility of the "
        "sources and methods that feed 'input_ids' (how openly stated, traceable, "
        "and parameterised they are in the paper) with (b) description coherence -- "
        "how well the figure/table's claim aligns with what those inputs can "
        "plausibly produce. 0 = inputs unavailable or claim does not follow from "
        "them; 100 = every input is fully reproducible AND the claim follows "
        "directly from them.",
    )
    reproducibility_rationale: str = Field(
        default="",
        description="Short justification (<=200 chars) for the assigned "
        "'reproducibility_score', citing the upstream gaps and any "
        "input-vs-claim coherence issues that drove the value.",
    )


class Hyperparameter(BaseModel):
    name: str
    value: str
    context: str = ""
    source_quote: str = ""


class SoftwareVersion(BaseModel):
    name: str
    version: str


class ProcessNodesResponse(BaseModel):
    nodes_process: list[ProcessNode]


class SourceNodesResponse(BaseModel):
    nodes_source: list[Dataset]


class ArtefactsResponse(BaseModel):
    repository_links: list[str] = Field(default_factory=list)
    supplementary_materials: list[str] = Field(default_factory=list)
    hardware_requirements: str | None = None


class SinkNodesResponse(BaseModel):
    nodes_sink: list[SinkNode] = Field(default_factory=list)


class ParametersResponse(BaseModel):
    software_versions: list[SoftwareVersion] = Field(default_factory=list)
    hyperparameters: list[Hyperparameter] = Field(default_factory=list)
    hardware: str | None = None


class PaperHeader(BaseModel):
    title: str = ""
    authors: list[str] = Field(default_factory=list)


class PaperMetadata(BaseModel):
    """Bibliographic + provenance + paper-level resources, kept apart from graph nodes."""

    pdf_path: str
    extraction_model: str = ""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    repository_links: list[str] = Field(default_factory=list)
    supplementary_materials: list[str] = Field(default_factory=list)
    hyperparameters: list[Hyperparameter] = Field(default_factory=list)
    stated_software_versions: dict[str, str] = Field(default_factory=dict)
    hardware_requirements: str | None = None


class PaperProfile(BaseModel):
    """Structured, source-grounded representation of a research paper."""

    metadata: PaperMetadata
    nodes_source: list[Dataset] = Field(default_factory=list)
    nodes_process: list[ProcessNode] = Field(default_factory=list)
    nodes_sink: list[SinkNode] = Field(default_factory=list)


SYSTEM_INSTRUCTION = (
    "You are analysing a research paper for reproducibility. "
    "Answer using ONLY information explicitly stated in the attached PDF. "
    "If a fact is not stated, use null / omit it -- do NOT infer, guess, or "
    "draw on outside knowledge. Every item you extract must include a short "
    "literal quote from the paper that grounds it. Respond with valid JSON "
    "conforming to the provided schema; return nothing else.\n\n"
    "BE TERSE. Avoid verbosity at all costs: do not restate the schema, do "
    "not add commentary, do not pad strings. Every string field is plain "
    "ASCII -- no combining diacritics, no non-BMP symbols, no long runs of "
    "repeated characters or tokens. Quotes are <=200 characters, "
    "descriptions <=400 characters, list items <=60 characters, lists "
    "<=8 items. If a field would exceed these limits, trim it; never "
    "produce filler to reach a limit. Close the JSON as soon as the "
    "required fields are populated -- truncation is a parse error."
)

PROMPT_HEADER = (
    "Extract the paper's title and author list. If either is not clearly "
    "printed on the first page, use an empty string or empty list."
)


def build_process_nodes_prompt(source_ids: list[str], sink_ids: list[str]) -> str:
    """Process-nodes prompt, parameterised with source/sink node_ids extracted upstream.

    Injecting the canonical IDs forces the LLM to reuse them verbatim in
    input_ids / outcomes, making the source→process→sink wiring resolvable
    by exact string match without a reconciliation pass.
    """
    src_block = "\n".join(f"  - {s}" for s in source_ids) or "  (none)"
    sink_block = "\n".join(f"  - {s}" for s in sink_ids) or "  (none)"
    return (
        "List every step of the research workflow in sequential execution "
        "order. Each step is a 'process node'; classify it as either a "
        "method (algorithmic / computational procedure) or an experiment "
        "(controlled trial, evaluation, ablation, comparison, sweep). "
        "Assign each step a canonical 'node_id' of the form 'meth_<NN>' "
        "with a zero-padded two-digit ordinal, starting at 'meth_01' and "
        "incrementing by 1 for each subsequent step. Also provide a short "
        "'node_name' (<=8 words) and a 'description' of what the step does. "
        "Include a short literal quote (source_quote) anchoring each step.\n\n"
        "OUTPUT-SIZE LIMITS (hard):\n"
        "- Every field is plain ASCII. Do NOT emit combining diacritics, "
        "  non-BMP symbols, or long runs of repeated characters.\n"
        "- 'source_quote': <=200 characters, a single literal sentence "
        "  fragment copied verbatim from the PDF.\n"
        "- 'description': <=400 characters.\n"
        "- 'node_name': <=8 words.\n"
        "- Each list ('input_ids', 'outcomes', 'tools_required', "
        "  'tools_mentioned', 'parameters_required', 'parameters_mentioned') "
        "  has at most 8 items; each item is <=60 characters.\n"
        "- Never repeat the same token more than twice in a row. If you are "
        "  near a field limit, stop and close the JSON -- truncation is a "
        "  parse error.\n\n"
        "FIELD SEMANTICS:\n"
        "- 'process_type': one of\n"
        "    * 'method'     -- an algorithmic or computational procedure "
        "       (e.g. preprocessing, feature extraction, model training, "
        "       inference, optimisation, simulation step);\n"
        "    * 'experiment' -- a controlled trial that produces measured "
        "       outcomes (e.g. evaluation on a held-out set, ablation "
        "       study, hyperparameter sweep, baseline comparison, "
        "       statistical test).\n"
        "- 'input_ids': labels consumed by this step (source ids, sink ids, "
        "  or verbatim labels emitted by earlier steps' 'outcomes').\n"
        "- 'outcomes': labels produced by this step. Most are INTERMEDIATE "
        "  artefacts consumed by a later step (partitions, preprocessed "
        "  data, features, trained models, prediction tensors, metrics, "
        "  etc.). Only the FINAL step(s) in a chain emit a sink id. A step "
        "  may emit multiple outcomes: any number of intermediates plus at "
        "  most one sink id.\n"
        "- 'tools_required': the GENERIC technique / capability needed to "
        "  execute the step (e.g. 'gradient-boosted classifier', 'stratified "
        "  k-fold cross-validation', '2-D convolutional encoder'). State "
        "  these even if the paper does not name a concrete library.\n"
        "- 'tools_mentioned': CONCRETE tools / libraries / frameworks the "
        "  paper explicitly names for this step, with version numbers when "
        "  stated (append 'version not stated' otherwise). Leave empty if "
        "  the paper does not name a tool for this step.\n"
        "- 'parameters_required': names of parameters the technique NEEDS "
        "  to be reproducible (e.g. 'learning_rate', 'batch_size', "
        "  'random_seed', 'n_estimators'), regardless of whether the paper "
        "  provides a value. Use lowercase snake_case parameter names.\n"
        "- 'parameters_mentioned': parameters the paper actually states for "
        "  this step, as 'name=value' strings (e.g. 'learning_rate=1e-3', "
        "  'n_folds=5'). Only include parameters whose value is explicitly "
        "  stated in the paper.\n"
        "  IMPORTANT: tables and figures that are NOT results (e.g. tables "
        "  of hyperparameters, configuration tables, architecture diagrams, "
        "  notation glossaries, dataset-statistics tables, methodology "
        "  flowcharts) are NOT nodes_sink -- they are part of the "
        "  methodology specification. MINE them: extract every parameter "
        "  value listed in such tables into 'parameters_mentioned' of the "
        "  step that uses the parameter, fold any algorithmic detail shown "
        "  in such figures into the step's 'description', and add any tool "
        "  / library named there to 'tools_mentioned'. Anchor each extracted "
        "  fact with a 'source_quote' from the table caption, table cell, "
        "  or figure caption.\n"
        "- 'algorithm_clarity': ordinal reconstructability score r(.) on a "
        "  4-level scale (integer in {1, 2, 3, 4}):\n"
        "    * 1 -- MISSING information: the paper names the step but provides "
        "         no algorithmic detail, no inputs, and no parameter values.\n"
        "    * 2 -- PARTIAL specification: the algorithm is named with some "
        "         detail, but at least one required input or parameter is "
        "         absent or only loosely described.\n"
        "    * 3 -- MOSTLY SPECIFIED components: most algorithmic choices, "
        "         inputs, and required parameters are stated; minor gaps "
        "         remain (e.g. an unstated default, an ambiguous tie-break).\n"
        "    * 4 -- SUFFICIENT detail for independent reconstruction: the "
        "         step's algorithm, inputs, and all required parameters are "
        "         stated precisely enough to reimplement without guessing.\n"
        "  Emit the BARE INTEGER (1, 2, 3, or 4) -- not a string.\n"
        "- 'reproducibility_score': INTEGER percentage in [0, 100] derived "
        "  from the four fields above, per this rubric:\n"
        "    * 0   -- 'algorithm_clarity'=1 AND no tools / parameters listed.\n"
        "    * 100 -- 'algorithm_clarity'=4 AND every item in 'tools_required' "
        "             also appears in 'tools_mentioned' AND every item in "
        "             'parameters_required' has a value in "
        "             'parameters_mentioned'.\n"
        "    * Intermediate values reflect partial specification. Suggested "
        "      anchors:\n"
        "        ~25 -- 'algorithm_clarity'=2 with most tools / parameters missing;\n"
        "        ~50 -- 'algorithm_clarity'=3 with about half the tools / "
        "               parameters named;\n"
        "        ~75 -- 'algorithm_clarity'=3-4 with most tools named and most "
        "               parameter values stated, only minor gaps remain.\n"
        "  Treat the score as the GEOMETRIC integration of (clarity / 4), "
        "  (tools_named_ratio), and (params_valued_ratio) -- if any of the "
        "  three is zero, the score should be near zero. Do not round to a "
        "  flat 50 when the inputs disagree.\n"
        "- 'reproducibility_rationale': short string (<=200 chars) citing the "
        "  concrete gaps that drove the score, e.g. 'clarity=3, 1/2 tools "
        "  named, 4/5 params stated' or 'clarity=4, all tools+params stated'.\n\n"
        "SOURCE IDS -- use these verbatim in 'input_ids' whenever a step "
        "consumes a raw dataset. Do NOT use human-readable dataset names, "
        "and do NOT invent new 'src_*' ids:\n"
        f"{src_block}\n\n"
        "SINK IDS -- use these verbatim in 'outcomes' ONLY when the step's "
        "direct output IS one of the figures/tables reported in the paper "
        "(typically the last step in a chain). If the step produces an "
        "intermediate artefact that a later step will turn into a figure "
        "or table, DO NOT put a sink id on this step. Do NOT invent new "
        "'sink_*' ids:\n"
        f"{sink_block}\n\n"
        "OUTCOME NAMING TEMPLATES (critical):\n"
        "When a step derives an artefact from a source dataset, name the "
        "outcome using these templates, where <slug> is the source node_id "
        "WITH THE 'src_' PREFIX REMOVED (e.g. for source 'src_adni_t1' use "
        "slug 'adni_t1'):\n"
        "  * 'train_<slug>', 'val_<slug>', 'test_<slug>' for splits\n"
        "  * 'preprocessed_<slug>' for cleaned / normalised data\n"
        "  * 'features_<slug>' for extracted features\n"
        "  * 'predictions_<slug>' for model outputs evaluated on that dataset\n"
        "  * 'metrics_<slug>' for evaluation metrics computed over that dataset\n"
        "If the paper uses a different but specific name for the artefact, "
        "you MAY use it; otherwise apply these templates so downstream "
        "steps can reference the artefact by a predictable name. Purely "
        "internal intermediates (not tied to any one source dataset) may "
        "use free-form lowercase snake_case names.\n\n"
        "LABEL CONSISTENCY RULES (critical):\n"
        "- Use the SAME label for the same artefact across steps. If step N "
        "  produces an outcome that feeds step N+1, the string in step N's "
        "  'outcomes' MUST appear verbatim in step N+1's 'input_ids'. Do "
        "  not paraphrase, pluralise, or reorder words between steps; pick "
        "  one canonical name per artefact and reuse it.\n"
        "- When a step splits, partitions, or divides a dataset (e.g. "
        "  train/test split, k-fold, train/val/test, stratified sampling), "
        "  'outcomes' MUST list each resulting partition as a separate "
        "  named item using the 'train_<slug>' / 'val_<slug>' / "
        "  'test_<slug>' templates above (use whatever names the paper uses "
        "  if stated). The 'description' MUST state the split percentages / "
        "  proportions / fold count exactly as reported in the paper (e.g. "
        "  '80/20 split', '70/15/15', '5-fold cross-validation'). "
        "  Downstream steps that consume a partition MUST reference it by "
        "  the same name in 'input_ids'."
    )


PROMPT_SOURCE_NODES = (
    "List every dataset referenced in this paper. For each dataset, emit an "
    "object with these fields (in this order):\n"
    "  * node_id                    -- canonical 'src_<slug>' id (rules below)\n"
    "  * node_name                  -- human-readable dataset name as printed in the paper\n"
    "  * source_quote               -- short literal quote (<=200 chars) anchoring the dataset\n"
    "  * description                -- description of contents (<=400 chars)\n"
    "  * size_estimate              -- approximate size (string) if stated, else null\n"
    "  * license                    -- licence string if stated, else null\n"
    "  * availability               -- 'open', 'upon_request', or 'not_mentioned'\n"
    "  * url                        -- URL or DOI if provided, else null\n"
    "  * reproducibility_score      -- INTEGER percentage in [0, 100] (rubric below)\n"
    "  * reproducibility_rationale  -- short justification (<=200 chars) for that score\n\n"
    "REPRODUCIBILITY SCORE RUBRIC (critical):\n"
    "- 0   -- the dataset is referenced but is NOT available: no URL, no DOI, no "
    "         repository, no acquisition path stated; or the paper marks it as "
    "         'not_mentioned'.\n"
    "- 100 -- the dataset is openly mentioned with a concrete, working source "
    "         (URL / DOI / public repository) that a third party can follow to "
    "         obtain the exact dataset.\n"
    "- Intermediate values reflect partial information, e.g.:\n"
    "    * ~25  -- only a citation to a prior paper, no URL or repository;\n"
    "    * ~50  -- access 'upon request' from authors, no public link;\n"
    "    * ~75  -- public repository named (e.g. 'available on Zenodo') but no "
    "             direct URL / DOI given.\n"
    "Always set 'reproducibility_rationale' to a short string citing the "
    "specific facts that drove the score (e.g. 'open URL provided', "
    "'citation-only, no DOI', 'upon-request, no link'). Anchor the score in "
    "what is actually stated in the paper -- do not infer availability from "
    "outside knowledge.\n\n"
    "NODE ID RULES (critical):\n"
    "- 'node_id' is used verbatim by nodes_process[*].input_ids to wire "
    "  sources into the workflow graph, so format and stability matter.\n"
    "- Format: 'src_<slug>' where <slug> is lowercase ASCII, digits and "
    "  underscores only, max 40 chars, derived from the most specific noun "
    "  phrase naming this dataset in the paper (NOT from description or "
    "  license). Examples: 'src_adni_t1', 'src_synthetic_reward_traces', "
    "  'src_mnist'. Do not include the year, version, or author name "
    "  unless the paper uses it as part of the dataset's primary name.\n"
    "- node_id MUST be unique across the returned list.\n"
    "- Keep node_id stable: for the same dataset in the same paper, the "
    "  same slug should be emitted every run."
)

PROMPT_ARTEFACTS = (
    "List all code repositories, supplementary materials, and external "
    "resources mentioned anywhere in this paper -- references, footnotes, "
    "acknowledgements, data availability statement. Include URLs, GitHub "
    "links, Zenodo DOIs, and institutional repository references. Also "
    "state any stated hardware requirements."
)


def build_sink_nodes_prompt(source_ids: list[str]) -> str:
    """Sink-nodes prompt, parameterised with source node_ids extracted upstream.

    Every figure and table is emitted as a unified SinkNode object with a
    'type' discriminator. Injecting source_ids forces input_ids to use them
    verbatim when an artefact draws directly on a raw dataset.
    """
    src_block = "\n".join(f"  - {s}" for s in source_ids) or "  (none)"
    return (
        "List ONLY the figures and tables that REPORT RESULTS of the study. "
        "Emit a sink_node ONLY for an artefact that presents an outcome, "
        "measurement, evaluation, comparison, or finding produced BY the "
        "paper's methodology -- the kind of artefact a replication would have "
        "to recompute to claim the result was reproduced.\n\n"
        "EXCLUDE (do NOT emit a sink_node for these):\n"
        "  * methodology / architecture diagrams, schematics, flowcharts, "
        "    pipelines, conceptual figures, illustrative cartoons;\n"
        "  * tables that list hyperparameters, configuration values, network "
        "    architecture, dataset statistics, parameter ranges, software "
        "    versions, hardware specifications, symbol/notation glossaries, "
        "    or experimental setup;\n"
        "  * tables of related work, summaries of prior literature, or "
        "    comparisons of methods that predate the paper's contribution;\n"
        "  * sample / qualitative example figures that show inputs rather "
        "    than outcomes; legend or colour-key panels.\n"
        "If unsure whether an artefact is a result, ask: 'Does this report a "
        "MEASURED OUTCOME of the study?' If no, EXCLUDE it. Information from "
        "excluded tables/figures (parameter values, methodology details) is "
        "captured downstream by the nodes_process extraction; do NOT duplicate "
        "it here.\n\n"
        "For each remaining results-bearing figure or table, emit a single "
        "sink_node object with these fields:\n"
        "  * node_id            -- canonical 'sink_fig<N>' or 'sink_tab<N>' (rules below)\n"
        "  * node_name          -- the paper's printed label, e.g. 'Figure 3', 'Table 1'\n"
        "  * source_quote       -- short literal quote (<=200 chars) from the caption or prose\n"
        "  * description        -- what the figure/table shows or presents (<=400 chars)\n"
        "  * input_ids          -- predecessor labels (rules below)\n"
        "  * size_estimate      -- approximate size if stated (e.g. 'N=120 cells x 6 columns', "
        "                          '8 panels'), else null\n"
        "  * type               -- 'figure' or 'table'\n"
        "  * statement_clarity  -- ordinal reconstructability score r(.) on a "
        "                          4-level scale (BARE INTEGER in {1, 2, 3, 4}):\n"
        "      1 -- MISSING information: only the artefact is shown; no procedure or "
        "         inputs are described.\n"
        "      2 -- PARTIAL: the procedure is named with some detail, but at least one "
        "         required input or step is absent or only loosely described.\n"
        "      3 -- MOSTLY SPECIFIED: most steps and inputs are stated; minor gaps "
        "         remain (e.g. an unstated default, an ambiguous parameter).\n"
        "      4 -- SUFFICIENT detail for independent reconstruction: every step "
        "         FROM 'input_ids' TO this artefact is stated precisely enough to "
        "         reimplement without guessing.\n"
        "  * statement_validity -- one of:\n"
        "      'supported'            -- claims directly justified by the listed inputs;\n"
        "      'partially_supported'  -- some claims justified, others not;\n"
        "      'unsupported'          -- conclusions exceed what the inputs can support;\n"
        "      'not_assessed'         -- insufficient information to judge.\n"
        "  * reproducibility_score      -- INTEGER percentage in [0, 100] (rubric below)\n"
        "  * reproducibility_rationale  -- short justification (<=200 chars) for that score\n\n"
        "REPRODUCIBILITY SCORE RUBRIC (critical):\n"
        "The score combines TWO factors:\n"
        "  (a) UPSTREAM REPRODUCIBILITY -- how openly stated, traceable, and "
        "      parameterised the sources and method steps feeding 'input_ids' are "
        "      (raw datasets cited with URL/DOI? methods specified clearly? "
        "      parameter values stated? tools named?). Treat this as the SUM / "
        "      AGGREGATE of the upstream nodes that produce the inputs to this "
        "      artefact: weak inputs cap the achievable score.\n"
        "  (b) DESCRIPTION COHERENCE -- how well the figure/table's stated "
        "      result follows from those inputs. A well-described result that "
        "      cleanly aggregates / visualises its inputs scores high; a "
        "      result that introduces unexplained data, missing intermediate "
        "      steps, or claims beyond what the inputs can yield scores low.\n"
        "Anchors:\n"
        "  * 0   -- inputs are unavailable / unspecified, OR the artefact's "
        "           claim does not follow from them.\n"
        "  * 100 -- every input is fully reproducible (open data, fully "
        "           specified methods) AND the claim follows directly and "
        "           coherently from those inputs.\n"
        "  * Intermediate values reflect partial information, e.g.:\n"
        "      ~25 -- some inputs cited but not retrievable; coherence weak;\n"
        "      ~50 -- about half of the upstream chain is reproducible and "
        "             the result is plausibly derivable;\n"
        "      ~75 -- inputs mostly reproducible and the result follows "
        "             clearly, with only minor unstated steps.\n"
        "Treat the score as a GEOMETRIC integration of (a) and (b) -- if either "
        "is near zero (unavailable inputs OR incoherent claim) the score should "
        "be near zero. Always set 'reproducibility_rationale' to a short string "
        "citing the specific upstream gaps and any input-vs-claim coherence "
        "issues that drove the value (e.g. 'inputs cited only, coherent claim', "
        "'inputs open + URL, claim aggregates them directly', 'one input "
        "missing, claim partially supported').\n\n"
        "NODE ID RULES (critical):\n"
        "- Each figure: 'sink_fig<N>' (e.g. 'sink_fig3', 'sink_fig3a' for sub-panels).\n"
        "- Each table:  'sink_tab<N>' (e.g. 'sink_tab1').\n"
        "- node_id MUST be unique across the returned list.\n\n"
        "INPUT ID RULES (critical):\n"
        "- 'input_ids' lists the immediate predecessor artefacts the figure or table is "
        "  built from (lists <=8 items, each <=60 characters).\n"
        "- Use the source ids below VERBATIM whenever the artefact draws directly on a raw "
        "  dataset.\n"
        "- Otherwise use a short lowercase snake_case label that a method step would "
        "  naturally emit (e.g. 'metrics_<dataset>', 'predictions_<dataset>', "
        "  'features_<dataset>'). Downstream method extraction will reuse these labels.\n\n"
        "SOURCE IDS -- use these verbatim in 'input_ids' whenever the figure/table "
        "consumes a raw dataset:\n"
        f"{src_block}\n\n"
        "OUTPUT-SIZE LIMITS (hard):\n"
        "- All strings plain ASCII; no combining diacritics or non-BMP symbols.\n"
        "- 'source_quote' <=200 chars; 'description' <=400 chars; 'node_name' <=8 words."
    )


PROMPT_PARAMETERS = (
    "List all parameters, hyperparameters, configuration values, and "
    "hardware requirements mentioned in this paper. Include numerical "
    "values, ranges, and any stated defaults. Return software versions as "
    "a list of objects, each with 'name' and 'version' fields."
)


# Module-level cache so the heavy LLM engine is loaded only once per process.
_VLLM_CACHE: dict[str, "LLM"] = {}
# vLLM's LLM.chat() is not thread-safe (ZMQ router assertion); serialize calls.
_VLLM_LOCK = threading.Lock()


def _pdf_to_markdown(pdf_path: Path) -> str:
    """Extract PDF text as markdown via MarkItDown (for text-only models)."""
    from markitdown import MarkItDown

    return MarkItDown().convert(str(pdf_path)).text_content


class VLLMOfflinePaperAnalyst:
    """Structured reproducibility extraction using vLLM offline inference.

    Loads the model once via vLLM's LLM engine (no HTTP server required) and
    runs the same six-query pipeline as the other analysts.  Guided JSON
    decoding (StructuredOutputsParams) enforces Pydantic schema conformance
    natively, so no post-processing is needed.

    The LLM engine is cached at module level so repeated instantiation (one per
    experiment run) does not reload from disk each time.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 65536,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        max_model_len: int | None = None,
    ) -> None:
        from vllm import LLM

        self.model_id = model
        self.temperature = float(temperature)
        self.max_tokens = max_tokens

        if model not in _VLLM_CACHE:
            kwargs: dict = dict(
                model=model,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=trust_remote_code,
            )
            if max_model_len is not None:
                kwargs["max_model_len"] = max_model_len
            _VLLM_CACHE[model] = LLM(**kwargs)
        self._llm = _VLLM_CACHE[model]

    def query(
        self,
        pdf_content: str,
        prompt: str,
        response_schema: type[T],
        query_name: str = "query",
    ) -> T:
        import json

        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        schema_str = json.dumps(response_schema.model_json_schema(), indent=2)
        full_prompt = (
            f"{prompt}\n\n"
            f"Respond with valid JSON that conforms exactly to this schema "
            f"(output the JSON object only, no prose):\n{schema_str}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"<paper>\n{pdf_content}\n</paper>\n\n{full_prompt}"},
        ]
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            structured_outputs=StructuredOutputsParams(
                json=response_schema.model_json_schema()
            ),
        )
        with _VLLM_LOCK:
            outputs = self._llm.chat(
                [messages],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        text = outputs[0].outputs[0].text
        return response_schema.model_validate_json(text)

    def analyze(self, pdf_path: str | Path, verbose: bool = True) -> PaperProfile:
        """Run the full six-query pipeline on a PDF and return a PaperProfile."""
        pdf_path = Path(pdf_path)
        if verbose:
            print(f"Preparing {pdf_path.name} for {self.model_id} ...")
        pdf_content = _pdf_to_markdown(pdf_path)
        if verbose:
            print(f"  extracted {len(pdf_content):,} chars of markdown")

        def run(name: str, prompt: str, schema: type[T]) -> T:
            if verbose:
                print(f"  querying: {name} ...", flush=True)
            return self.query(pdf_content, prompt, schema, query_name=name)

        header = run("header", PROMPT_HEADER, PaperHeader)
        sources = run("nodes_source", PROMPT_SOURCE_NODES, SourceNodesResponse)
        source_ids = [d.node_id for d in sources.nodes_source]
        sinks = run("nodes_sink", build_sink_nodes_prompt(source_ids), SinkNodesResponse)
        sink_ids = [s.node_id for s in sinks.nodes_sink]
        processes = run(
            "nodes_process",
            build_process_nodes_prompt(source_ids, sink_ids),
            ProcessNodesResponse,
        )
        artefacts = run("artefacts", PROMPT_ARTEFACTS, ArtefactsResponse)
        params = run("parameters", PROMPT_PARAMETERS, ParametersResponse)

        metadata = PaperMetadata(
            pdf_path=str(pdf_path),
            extraction_model=self.model_id,
            title=header.title,
            authors=header.authors,
            repository_links=artefacts.repository_links,
            supplementary_materials=artefacts.supplementary_materials,
            hyperparameters=params.hyperparameters,
            stated_software_versions={sv.name: sv.version for sv in params.software_versions},
            hardware_requirements=artefacts.hardware_requirements or params.hardware,
        )
        return PaperProfile(
            metadata=metadata,
            nodes_source=sources.nodes_source,
            nodes_process=processes.nodes_process,
            nodes_sink=sinks.nodes_sink,
        )
