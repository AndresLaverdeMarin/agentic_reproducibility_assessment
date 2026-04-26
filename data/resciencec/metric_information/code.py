import os
import json
import time
import pandas as pd
from pathlib import Path
from google import genai
from google.genai import types
import subprocess
from pathlib import Path

API_KEY = os.environ["GEMINI_API_KEY"]

ORIGINAL_DIR = Path("../original_pdfs")
RESCIENCE_DIR = Path("../rescience_pdfs")
REPAIRED_DIR = Path("../repaired_pdfs")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

MODEL = "gemini-2.5-flash"   # cheaper
# MODEL = "gemini-2.5-pro"   # stronger, more expensive

client = genai.Client(api_key=API_KEY)

f = open("prompt.txt", "r")
PROMPT = f.read()
f.close()

SCHEMA = {
    "type": "object",
    "properties": {
        "metadata": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"}
            },
            "required": ["filename"]
        },
        "scores": {
            "type": "object",
            "properties": {
                "Sources": {"type": "integer"},
                "Methods": {"type": "integer"},
                "Experiments": {"type": "integer"},
                "Sinks": {"type": "integer"}
            },
            "required": ["Sources", "Methods", "Experiments", "Sinks"]
        },
        "confidence": {
            "type": "object",
            "properties": {
                "Sources": {"type": "integer"},
                "Methods": {"type": "integer"},
                "Experiments": {"type": "integer"},
                "Sinks": {"type": "integer"}
            },
            "required": ["Sources", "Methods", "Experiments", "Sinks"]
        },
        "reasoning": {
            "type": "object",
            "properties": {
                "Sources": {"type": "string"},
                "Methods": {"type": "string"},
                "Experiments": {"type": "string"},
                "Sinks": {"type": "string"}
            },
            "required": ["Sources", "Methods", "Experiments", "Sinks"]
        }
    },
    "required": ["metadata", "scores", "confidence", "reasoning"]
}

def upload_pdf(path: Path):
    return client.files.upload(file=str(path))

def assess_pair(filename: str):
    original_path = ORIGINAL_DIR / filename
    rescience_path = RESCIENCE_DIR / filename

    print("\tuploading original_file...")
    original_file = upload_pdf(original_path)
    print("\tuploading rescience_file...")
    rescience_file = upload_pdf(rescience_path)
    print("\tuploads done")

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            "ORIGINAL_PAPER:",
            original_file,
            "REPRODUCTION_REPORT:",
            rescience_file,
            PROMPT,
            f"Filename: {filename}"
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=SCHEMA,
        ),
    )

    result = json.loads(response.text)
    result["filename"] = filename
    return result

def main():
    files = sorted(p.name for p in ORIGINAL_DIR.glob("*.pdf"))

    results = []

    for i, filename in enumerate(files, start=1):
        json_filename = filename.replace(".pdf", ".json")
        json_path = OUT_DIR / json_filename

        # skip already processed papers
        if json_path.exists():
            print(f"[{i}/{len(files)}] skipping {filename}")
            continue

        if not (RESCIENCE_DIR / filename).exists():
            print(f"[{i}/{len(files)}] missing reproduction report: {filename}")
            continue

        print(f"[{i}/{len(files)}] scoring {filename}")

        try:
            result = assess_pair(filename)

            # store per-paper JSON file
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            results.append(result)

        except Exception as e:
            print(f"ERROR on {filename}: {e}")

        time.sleep(2)

    # rebuild dataframe from all JSON files
    all_results = []

    for json_file in sorted(OUT_DIR.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            all_results.append(json.load(f))

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUT_DIR / "scores.csv", index=False)
        print(f"\nSaved CSV summary to {OUT_DIR / 'scores.csv'}")

if __name__ == "__main__":
    main()