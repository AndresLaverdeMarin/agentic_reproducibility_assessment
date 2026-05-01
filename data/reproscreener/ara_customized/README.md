# ARA Customized for ReproScreener

This folder contains a separate ARA variant for the ReproScreener papers. It
does not modify `src/ara_pipeline/online_llm_pipelines.py`.

`ara_customized_pipeline.py` imports the canonical ARA prompts and schemas,
runs the same six ARA extraction queries, and adds a seventh structured query
for:

- `problem`
- `objective`
- `research_method`
- `research_questions`
- `dataset`
- `hypothesis`
- `prediction`
- `code_available`
- `experiment_setup`

Each category is emitted as:

```json
{
  "assessment": 1,
  "confidence": 0.9,
  "reasoning": "Brief evidence or missing-item rationale."
}
```

`ara_executor.py` is intended to be run directly from Spyder with Gemini only.
Put a `.env` file next to it with `GEMINI_API_KEY` or `GOOGLE_API_KEY`, review
the configuration constants at the top of the file, and run it. It defaults to
`gemini-3.1-pro`. Outputs are written to `outputs/` in this folder.
