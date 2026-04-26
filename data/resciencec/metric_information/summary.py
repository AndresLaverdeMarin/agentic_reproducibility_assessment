import csv
import json
from pathlib import Path


OUTPUTS_DIR = Path("outputs")
CSV_PATH = Path("scores.csv")
DIMENSIONS = ["Sources", "Methods", "Experiments", "Sinks"]


def load_result(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_row(data: dict, fallback_name: str) -> dict:
    metadata = data.get("metadata", {})
    scores = data.get("scores", {})
    confidence = data.get("confidence", {})

    row = {
        "paper": data.get("filename") or fallback_name,
    }

    for dimension in DIMENSIONS:
        row[f"score_{dimension}"] = scores.get(dimension, "")
        row[f"confidence_{dimension}"] = confidence.get(dimension, "")

    return row


def main() -> None:
    json_files = sorted(
        path for path in OUTPUTS_DIR.glob("*.json") if path.is_file()
    )

    fieldnames = ["paper"]
    for dimension in DIMENSIONS:
        fieldnames.append(f"score_{dimension}")
    for dimension in DIMENSIONS:
        fieldnames.append(f"confidence_{dimension}")

    rows = []
    for json_path in json_files:
        data = load_result(json_path)
        rows.append(build_row(data, json_path.stem))

    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
