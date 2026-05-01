"""Gemini-backed replicability scorer for paired (original, human-reproduction) PDFs.

Companion to ``src/ara_pipeline/online_llm_pipelines.py``: that module extracts a workflow
profile from a single paper; this one scores how reproducible an ORIGINAL_PAPER
proved to be, given a REPRODUCTION_REPORT as evidence. Four dimension-level
1-4 ordinal scores (``Sources`` / ``Methods`` / ``Experiments`` / ``Sinks``)
plus one overall 1-4 ordinal score, each with a 0-100 confidence and a
free-text reasoning string. Pydantic schema, prompt, pair
upload, and the resume-on-existing-JSON sweep are inlined so the file is
self-contained -- mirroring the GeminiPaperAnalyst surface.

Field names ``Sources/Methods/Experiments/Sinks`` are intentionally capitalised
(not snake_case) so the produced JSON matches the legacy output shape under
``outputs/`` and the existing ``summary.py`` / ``visualization.py`` consumers.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


# -- prompt -------------------------------------------------------------------

REPLICABILITY_PROMPT = (
    "You are assessing scientific reproducibility.\n\n"
    "You receive two PDFs:\n"
    "1. ORIGINAL_PAPER: the original scientific work.\n"
    "2. REPRODUCTION_REPORT: a human reproduction attempt of the original work.\n\n"
    "Your task is to score how reproducible the ORIGINAL_PAPER was, using the "
    "REPRODUCTION_REPORT as evidence of what human reproducers could or could not "
    "reconstruct.\n\n"
    "Score four dimensions from 1 to 4:\n\n"
    "Sources:\n"
    "Evaluates whether datasets, assumptions, and research questions are sufficiently "
    "specified to initialize the computational workflow.\n\n"
    "Methods:\n"
    "Evaluates whether intermediate transformations, algorithms, variables, "
    "parameterizations, software & hardware resources, and modeling steps are described "
    "clearly enough to permit independent implementation.\n\n"
    "Experiments:\n"
    "Evaluates whether evaluation protocols, baselines, runtime conditions, software & "
    "hardware resources, and experimental configurations are reproducible.\n\n"
    "Sinks:\n"
    "Evaluates whether figures, tables, and reported conclusions can be traced to "
    "explicitly described workflow steps and supporting evidence.\n\n"
    "Scale:\n"
    "1 = missing information\n"
    "2 = partial specification\n"
    "3 = mostly specified components\n"
    "4 = sufficient detail for independent reconstruction\n\n"
    "Please also provide one final score for the overall reproducibility of the paper from 1 to 4 (considering all four dimensions).\n"
    "Important rules:\n"
    "- Base the score primarily on evidence from the reproduction report, but check "
    "the original paper.\n"
    "- Penalize cases where reproducers had to guess, contact authors, reverse-engineer "
    "code, infer parameters, authors reproduce their own paper, or use undocumented "
    "assumptions.\n"
    "- Do not reward successful reproduction alone; score documentation quality and "
    "reconstructability.\n"
    "- If evidence is unclear, choose the lower score.\n"
    "- Return only valid JSON, confidence can be 0-100, score as specified in scale, "
    "and reasoning a text.\n"
)


# -- schema -------------------------------------------------------------------


class FacetScore(BaseModel):
    Sources: int = Field(ge=1, le=4)
    Methods: int = Field(ge=1, le=4)
    Experiments: int = Field(ge=1, le=4)
    Sinks: int = Field(ge=1, le=4)
    Overall: int = Field(ge=1, le=4)


class FacetConfidence(BaseModel):
    Sources: int = Field(ge=0, le=100)
    Methods: int = Field(ge=0, le=100)
    Experiments: int = Field(ge=0, le=100)
    Sinks: int = Field(ge=0, le=100)
    Overall: int = Field(ge=0, le=100)


class FacetReasoning(BaseModel):
    Sources: str
    Methods: str
    Experiments: str
    Sinks: str
    Overall: str


class ReplicabilityMetadata(BaseModel):
    filename: str


class ReplicabilityAssessment(BaseModel):
    metadata: ReplicabilityMetadata
    scores: FacetScore = Field(
        description="Ordinal reconstructability per facet and overall in {1,2,3,4}: "
        "1=missing, 2=partial, 3=mostly specified, "
        "4=sufficient for independent reconstruction."
    )
    confidence: FacetConfidence = Field(
        description="Confidence in each facet score, 0..100 (higher = more certain)."
    )
    reasoning: FacetReasoning = Field(
        description="Free-text justification per facet, citing specific evidence "
        "from the reproduction report and/or the original paper."
    )


# -- pricing ------------------------------------------------------------------

GEMINI_PRICING_USD_PER_MTOK = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
}


# -- analyst ------------------------------------------------------------------


class ReplicabilityPairAnalyst:
    """Score reproducibility for a paired (original_paper, reproduction_report) PDF set.

    Mirrors GeminiPaperAnalyst's surface: ``upload_pdf`` / ``assess`` / ``usage_summary``
    -- but takes two PDFs per call and emits a ``ReplicabilityAssessment``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
    ) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "Set GEMINI_API_KEY (or GOOGLE_API_KEY) or pass api_key= explicitly."
            )
        self.client = genai.Client(api_key=key)
        self.model = model
        self.temperature = float(temperature)
        self.usage_log: list[tuple[str, types.GenerateContentResponseUsageMetadata]] = []

    def upload_pdf(self, pdf_path: str | Path) -> types.File:
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return self.client.files.upload(file=str(path))

    def assess(
        self,
        original_pdf: str | Path,
        reproduction_pdf: str | Path,
        filename: str | None = None,
        verbose: bool = False,
    ) -> ReplicabilityAssessment:
        original_pdf = Path(original_pdf)
        reproduction_pdf = Path(reproduction_pdf)
        label = filename or original_pdf.name
        if verbose:
            print(f"  uploading original: {original_pdf.name}", flush=True)
        original_file = self.upload_pdf(original_pdf)
        if verbose:
            print(f"  uploading reproduction: {reproduction_pdf.name}", flush=True)
        reproduction_file = self.upload_pdf(reproduction_pdf)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                "ORIGINAL_PAPER:",
                original_file,
                "REPRODUCTION_REPORT:",
                reproduction_file,
                REPLICABILITY_PROMPT,
                f"Filename: {label}",
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReplicabilityAssessment,
                temperature=self.temperature,
                max_output_tokens=8192,
            ),
        )
        if response.usage_metadata is not None:
            self.usage_log.append(("assess", response.usage_metadata))
        finish = (
            getattr(response.candidates[0], "finish_reason", None) if response.candidates else None
        )
        if finish is not None and str(finish).upper().endswith("MAX_TOKENS"):
            raise RuntimeError(
                "assess: Gemini hit max_output_tokens before closing JSON -- "
                "raise max_output_tokens or tighten the prompt."
            )
        return ReplicabilityAssessment.model_validate_json(response.text)

    def usage_summary(self) -> dict:
        """Aggregate tokens + USD estimate over every call since the last reset."""
        input_tokens = sum((u.prompt_token_count or 0) for _, u in self.usage_log)
        output_tokens = sum((u.candidates_token_count or 0) for _, u in self.usage_log)
        rates = GEMINI_PRICING_USD_PER_MTOK.get(self.model)
        input_usd = input_tokens / 1_000_000 * rates["input"] if rates else None
        output_usd = output_tokens / 1_000_000 * rates["output"] if rates else None
        total_usd = input_usd + output_usd if rates else None
        return {
            "model": self.model,
            "calls": len(self.usage_log),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_usd": input_usd,
            "output_usd": output_usd,
            "total_usd": total_usd,
        }


def assess_pair(
    original_pdf: str | Path,
    reproduction_pdf: str | Path,
    filename: str | None = None,
    api_key: str | None = None,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
) -> ReplicabilityAssessment:
    """One-shot wrapper: paired PDFs -> ReplicabilityAssessment."""
    analyst = ReplicabilityPairAnalyst(api_key=api_key, model=model, temperature=temperature)
    return analyst.assess(original_pdf, reproduction_pdf, filename=filename)


# -- driver -------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ORIGINAL_DIR = HERE.parent / "papers_original"
REPRODUCTION_DIR = HERE.parent / "papers_humanreplication"
OUT_DIR = HERE / "outputs"
N_THREADS = 5
SUBMISSION_STAGGER_SECONDS = 0.5
OVERWRITE_EXISTING = False


def add_usage(total: dict, summary: dict) -> None:
    for key in ("calls", "input_tokens", "output_tokens", "total_tokens"):
        total[key] += summary[key]
    for key in ("input_usd", "output_usd", "total_usd"):
        if summary[key] is not None:
            total[key] += summary[key]


def score_one(filename: str) -> tuple[str, str, dict | None]:
    json_path = OUT_DIR / filename.replace(".pdf", ".json")
    reproduction_pdf = REPRODUCTION_DIR / filename

    try:
        analyst = ReplicabilityPairAnalyst(model="gemini-2.5-flash", temperature=0.0)
        result = analyst.assess(ORIGINAL_DIR / filename, reproduction_pdf, filename=filename)
        tmp_path = json_path.with_name(json_path.name + ".tmp")
        tmp_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(json_path)
        return filename, "done", analyst.usage_summary()
    except Exception as exc:
        return filename, f"ERROR {type(exc).__name__}: {exc}", None


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)

    files = sorted(p.name for p in ORIGINAL_DIR.glob("*.pdf"))
    if not files:
        print(f"No PDFs in {ORIGINAL_DIR}", flush=True)
        return 0

    pending = []
    for filename in files:
        json_path = OUT_DIR / filename.replace(".pdf", ".json")
        if json_path.exists() and not OVERWRITE_EXISTING:
            print(f"skip existing {filename}", flush=True)
            continue
        reproduction_pdf = REPRODUCTION_DIR / filename
        if not reproduction_pdf.is_file():
            print(f"missing reproduction report: {filename}", flush=True)
            continue
        pending.append(filename)

    if not pending:
        print("No files to score. Existing JSON files were skipped.", flush=True)
        return 0

    print(
        f"Scoring {len(pending)} files with N_THREADS={N_THREADS}. "
        "If Gemini rate limits this run, lower N_THREADS or increase "
        "SUBMISSION_STAGGER_SECONDS.",
        flush=True,
    )

    total_usage = {
        "model": "gemini-2.5-flash",
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "input_usd": 0.0,
        "output_usd": 0.0,
        "total_usd": 0.0,
    }

    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = []
        for filename in pending:
            futures.append(executor.submit(score_one, filename))
            time.sleep(SUBMISSION_STAGGER_SECONDS)

        for i, future in enumerate(as_completed(futures), start=1):
            filename, status, usage = future.result()
            print(f"[{i}/{len(pending)}] {status}: {filename}", flush=True)
            if usage is not None:
                add_usage(total_usage, usage)

    print("\n" + json.dumps(total_usage, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
