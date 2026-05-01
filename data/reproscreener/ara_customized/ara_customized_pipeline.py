"""ARA pipeline variant with ReproScreener category labels.

The canonical pipeline in ``src/ara_pipeline/online_llm_pipelines.py`` is not
modified. This module imports its schemas/prompts, runs the same extraction
queries, and adds one extra structured query that emits binary ReproScreener
category assessments.
"""

from __future__ import annotations

import sys
import types as py_types
from pathlib import Path
from typing import TypeVar

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is optional at import time
    load_dotenv = None


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if load_dotenv is not None:
    load_dotenv(THIS_DIR / ".env")
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(SRC_DIR / "ara_pipeline" / ".env")

# The canonical ARA module imports OpenAI symbols at module import time even
# when only the Gemini analyst is used. Some environments have no OpenAI SDK,
# or an older SDK without openai.types.responses. This shim keeps the custom
# Gemini-only pipeline independent of OpenAI while leaving the canonical module
# untouched.
if "openai" not in sys.modules:
    openai_stub = py_types.ModuleType("openai")

    class _OpenAIUnavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "OpenAI is not installed in this environment. "
                "Use CustomizedGeminiPaperAnalyst for this custom pipeline."
            )

    openai_stub.OpenAI = _OpenAIUnavailable
    sys.modules["openai"] = openai_stub
elif not hasattr(sys.modules["openai"], "OpenAI"):
    class _OpenAIUnavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "OpenAI is not available in this environment. "
                "Use CustomizedGeminiPaperAnalyst for this custom pipeline."
            )

    sys.modules["openai"].OpenAI = _OpenAIUnavailable

if "openai.types" not in sys.modules:
    sys.modules["openai.types"] = py_types.ModuleType("openai.types")

if "openai.types.responses" not in sys.modules:
    responses_stub = py_types.ModuleType("openai.types.responses")

    class _OpenAIResponseStub:
        pass

    responses_stub.Response = _OpenAIResponseStub
    sys.modules["openai.types.responses"] = responses_stub

from google.genai import types
from pydantic import BaseModel, Field

from ara_pipeline.online_llm_pipelines import (
    Dataset,
    GEMINI_PRICING_USD_PER_MTOK,
    GeminiPaperAnalyst,
    PaperHeader,
    PaperMetadata,
    PROMPT_ARTEFACTS,
    PROMPT_HEADER,
    PROMPT_PARAMETERS,
    PROMPT_SOURCE_NODES,
    ArtefactsResponse,
    ParametersResponse,
    ProcessNode,
    ProcessNodesResponse,
    SinkNode,
    SinkNodesResponse,
    SourceNodesResponse,
    build_process_nodes_prompt,
    build_sink_nodes_prompt,
)

T = TypeVar("T", bound=BaseModel)


REPROSCREENER_RUBRIC_PATH = (
    REPO_ROOT / "data" / "reproscreener" / "metric_information" / "raghupathi-2022" / "main.md"
)

REPROSCREENER_CATEGORY_DEFINITIONS = {
    "problem": "Problem statement: explicit mention of the problem the research seeks to address.",
    "objective": "Goal: explicit mention of the research goal or objective.",
    "research_method": "Research method: explicit mention of the research method used.",
    "research_questions": "Research question: explicit mention of the research question(s) addressed.",
    "dataset": "Dataset/data source: explicit mention of the data source, dataset, or data used.",
    "hypothesis": "Hypothesis: explicit mention of hypotheses being investigated.",
    "prediction": "Prediction: explicit mention of prediction related to the hypotheses.",
    "code_available": "Method source code: explicit statement that the system/method code is available as open source or otherwise accessible.",
    "experiment_setup": "Experiment setup: explicit sharing of variable settings such as hyperparameters, configuration values, or setup details.",
}

REPROSCREENER_SYSTEM_INSTRUCTION = (
    "You are analysing a research paper for ReproScreener reproducibility labels. "
    "Answer using ONLY information explicitly stated in the attached PDF. "
    "If a category is not explicitly stated, set assessment to 0; do NOT infer, "
    "guess, or draw on outside knowledge. Respond with valid JSON conforming to "
    "the provided schema; return nothing else.\n\n"
    "BE TERSE. Every string field is plain ASCII. Keep reasoning <=300 characters "
    "per category. The reasoning field is the grounding field: briefly state the "
    "explicit evidence when assessment=1, or the absence that drove assessment=0."
)


class ReproScreenerCategoryAssessment(BaseModel):
    assessment: int = Field(
        ge=0,
        le=1,
        description="Binary label: 1 if explicitly present in the PDF, otherwise 0.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the binary label, from 0.0 to 1.0.",
    )
    reasoning: str = Field(
        description=(
            "Brief reasoning grounded only in the paper. If assessment is 1, mention the "
            "explicit evidence; if 0, mention what was missing."
        )
    )


class ReproScreenerCategories(BaseModel):
    problem: ReproScreenerCategoryAssessment
    objective: ReproScreenerCategoryAssessment
    research_method: ReproScreenerCategoryAssessment
    research_questions: ReproScreenerCategoryAssessment
    dataset: ReproScreenerCategoryAssessment
    hypothesis: ReproScreenerCategoryAssessment
    prediction: ReproScreenerCategoryAssessment
    code_available: ReproScreenerCategoryAssessment
    experiment_setup: ReproScreenerCategoryAssessment


class CustomizedPaperProfile(BaseModel):
    """Canonical ARA profile plus top-level ReproScreener category labels."""

    metadata: PaperMetadata
    reproscreener_categories: ReproScreenerCategories
    nodes_source: list[Dataset] = Field(default_factory=list)
    nodes_process: list[ProcessNode] = Field(default_factory=list)
    nodes_sink: list[SinkNode] = Field(default_factory=list)


def _load_reproscreener_rubric() -> str:
    if REPROSCREENER_RUBRIC_PATH.is_file():
        return REPROSCREENER_RUBRIC_PATH.read_text(encoding="utf-8")
    return "\n".join(f"- {k}: {v}" for k, v in REPROSCREENER_CATEGORY_DEFINITIONS.items())


def build_reproscreener_categories_prompt() -> str:
    definitions = "\n".join(
        f"- {key}: {definition}" for key, definition in REPROSCREENER_CATEGORY_DEFINITIONS.items()
    )
    rubric = _load_reproscreener_rubric()
    return (
        "Assess the attached PDF using the Raghupathi 2022 ReproScreener rubric excerpt below. "
        "Return ONLY the nine requested categories in the provided JSON schema.\n\n"
        "Use strict binary labels:\n"
        "- assessment=1 only when the paper explicitly states the relevant item.\n"
        "- assessment=0 when the item is absent, only implicit, or would require inference.\n"
        "- confidence is a number from 0.0 to 1.0.\n"
        "- reasoning is a short plain-ASCII text grounded only in the PDF. Include a brief "
        "evidence phrase when present; do not invent quotes or outside facts.\n\n"
        "Requested output categories and their intended rubric mapping:\n"
        f"{definitions}\n\n"
        "Raghupathi 2022 rubric excerpt:\n"
        f"{rubric}"
    )


class CustomizedGeminiPaperAnalyst(GeminiPaperAnalyst):
    """Gemini-backed canonical ARA extraction plus ReproScreener labels."""

    usage_log: list[tuple[str, types.GenerateContentResponseUsageMetadata]]

    def query_reproscreener(self, file_handle: types.File) -> ReproScreenerCategories:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[file_handle, build_reproscreener_categories_prompt()],
            config=types.GenerateContentConfig(
                system_instruction=REPROSCREENER_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=ReproScreenerCategories,
                temperature=self.temperature,
                max_output_tokens=8192,
            ),
        )
        if response.usage_metadata is not None:
            self.usage_log.append(("reproscreener_categories", response.usage_metadata))
        finish = (
            getattr(response.candidates[0], "finish_reason", None) if response.candidates else None
        )
        if finish is not None and str(finish).upper().endswith("MAX_TOKENS"):
            raise RuntimeError(
                "reproscreener_categories: Gemini hit max_output_tokens before closing JSON."
            )
        return ReproScreenerCategories.model_validate_json(response.text)

    def analyze(self, pdf_path: str | Path, verbose: bool = True) -> CustomizedPaperProfile:
        pdf_path = Path(pdf_path)
        self.usage_log = []
        if verbose:
            print(f"Uploading {pdf_path.name} to Gemini Files API...")
        fh = self.upload_pdf(pdf_path)

        def run(name: str, prompt: str, schema: type[T]) -> T:
            if verbose:
                print(f"  querying: {name} ...", flush=True)
            return self.query(fh, prompt, schema, query_name=name)

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
        if verbose:
            print("  querying: reproscreener_categories ...", flush=True)
        reproscreener = self.query_reproscreener(fh)

        metadata = PaperMetadata(
            pdf_path=str(pdf_path),
            extraction_model=self.model,
            title=header.title,
            authors=header.authors,
            repository_links=artefacts.repository_links,
            supplementary_materials=artefacts.supplementary_materials,
            hyperparameters=params.hyperparameters,
            stated_software_versions={sv.name: sv.version for sv in params.software_versions},
            hardware_requirements=artefacts.hardware_requirements or params.hardware,
        )
        return CustomizedPaperProfile(
            metadata=metadata,
            reproscreener_categories=reproscreener,
            nodes_source=sources.nodes_source,
            nodes_process=processes.nodes_process,
            nodes_sink=sinks.nodes_sink,
        )


def make_customized_analyst(
    model: str = "gemini-3.1-pro",
    temperature: float | None = 0.0,
) -> CustomizedGeminiPaperAnalyst:
    if model.lower().startswith(("gpt-", "o1", "o3", "o4")):
        raise ValueError("This customized pipeline is Gemini-only. Use a Gemini model name.")
    return CustomizedGeminiPaperAnalyst(
        model=model,
        temperature=0.0 if temperature is None else float(temperature),
    )


def analyze_pdf_customized(
    pdf_path: str | Path,
    model: str = "gemini-3.1-pro",
    temperature: float | None = 0.0,
) -> CustomizedPaperProfile:
    return make_customized_analyst(model=model, temperature=temperature).analyze(pdf_path)


def pricing_for_model(model: str) -> dict[str, float] | None:
    return GEMINI_PRICING_USD_PER_MTOK.get(model)
