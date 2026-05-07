"""Print the benchmark comparison table as Markdown."""

import csv
import json
import zipfile
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]

ROWS = ("ARA", "ReplicatorAgent", "ReproScreener")
DATASETS = ("Rescience C", "ReproBench", "GoldStandardDB")
DATASET_METRICS = {
    "Rescience C": ("ACC", "F1", "Distance", "Abs Distance"),
    "GoldStandardDB": ("ACC", "F1"),
    "ReproBench": ("ACC", "F1", "Distance", "Abs Distance"),
}
RESCIENCE_DIMENSIONS = (
    ("Sources", "R_sources"),
    ("Methods", "R_methods"),
    ("Experiments", "R_experiments"),
    ("Sinks", "R_sinks"),
    ("Overall", "R_0_4"),
)


def mean_std(values: list[float]) -> str:
    if not values:
        return "-"
    std = stdev(values) if len(values) > 1 else 0.0
    return f"{mean(values) * 100:.2f} +/- {std * 100:.2f}"


def mean_std_raw(values: list[float]) -> str:
    if not values:
        return "-"
    std = stdev(values) if len(values) > 1 else 0.0
    return f"{mean(values):.2f} +/- {std:.2f}"


def fmt_percent(value: float) -> str:
    return f"{value * 100:.2f}"


def macro_classification_scores(
    true_values: list[int],
    pred_values: list[int],
) -> tuple[list[float], list[float]]:
    if not true_values:
        return [], []

    classes = sorted(set(true_values) | set(pred_values))
    accuracies = []
    f1_scores = []

    for cls in classes:
        true_binary = [value == cls for value in true_values]
        pred_binary = [value == cls for value in pred_values]

        correct = sum(true == pred for true, pred in zip(true_binary, pred_binary))
        true_positives = sum(true and pred for true, pred in zip(true_binary, pred_binary))
        false_positives = sum((not true) and pred for true, pred in zip(true_binary, pred_binary))
        false_negatives = sum(true and (not pred) for true, pred in zip(true_binary, pred_binary))

        accuracies.append(correct / len(true_binary))
        denominator = (2 * true_positives) + false_positives + false_negatives
        f1_scores.append((2 * true_positives / denominator) if denominator else 0.0)

    return accuracies, f1_scores


def multiclass_scores(true_values: list[int], pred_values: list[int]) -> tuple[float, float]:
    """Direct pooled multiclass accuracy and macro-F1."""
    if not true_values:
        return 0.0, 0.0

    accuracy = sum(true == pred for true, pred in zip(true_values, pred_values)) / len(true_values)

    classes = sorted(set(true_values) | set(pred_values))
    f1_scores = []

    for cls in classes:
        true_positives = sum(
            true == cls and pred == cls
            for true, pred in zip(true_values, pred_values)
        )
        false_positives = sum(
            true != cls and pred == cls
            for true, pred in zip(true_values, pred_values)
        )
        false_negatives = sum(
            true == cls and pred != cls
            for true, pred in zip(true_values, pred_values)
        )

        denominator = (2 * true_positives) + false_positives + false_negatives
        f1_scores.append((2 * true_positives / denominator) if denominator else 0.0)

    return accuracy, mean(f1_scores)


def to_ordinal_class(value: float) -> int:
    return min(4, max(1, round(value)))


def unit_to_ordinal_class(value: float) -> int:
    return to_ordinal_class(1.0 + (3.0 * value))


def binary_scores(true_values: list[int], pred_values: list[int]) -> tuple[float, float]:
    if not true_values:
        return 0.0, 0.0

    correct = sum(true == pred for true, pred in zip(true_values, pred_values))
    true_positives = sum(true == 1 and pred == 1 for true, pred in zip(true_values, pred_values))
    false_positives = sum(true == 0 and pred == 1 for true, pred in zip(true_values, pred_values))
    false_negatives = sum(true == 1 and pred == 0 for true, pred in zip(true_values, pred_values))

    accuracy = correct / len(true_values)
    denominator = (2 * true_positives) + false_positives + false_negatives
    f1 = (2 * true_positives / denominator) if denominator else 0.0
    return accuracy, f1


def load_ara_reproscreener_outputs(
    outputs_dir: Path,
    outputs_zip: Path,
) -> dict[str, dict]:
    rows = {}
    json_paths = sorted(outputs_dir.glob("*.profile.json")) if outputs_dir.is_dir() else []

    if json_paths:
        for path in json_paths:
            paper_id = path.name.split("_T", maxsplit=1)[0]
            data = json.loads(path.read_text(encoding="utf-8"))
            rows[paper_id] = data.get("reproscreener_categories", {})
        return rows

    if not outputs_zip.is_file():
        return rows

    with zipfile.ZipFile(outputs_zip) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".profile.json"):
                continue
            paper_id = Path(name).name.split("_T", maxsplit=1)[0]
            with archive.open(name) as handle:
                data = json.loads(handle.read().decode("utf-8"))
            rows[paper_id] = data.get("reproscreener_categories", {})

    return rows


def load_rescience_truth_outputs(outputs_dir: Path, outputs_zip: Path) -> dict[str, dict]:
    rows = {}
    json_paths = sorted(outputs_dir.glob("*.json")) if outputs_dir.is_dir() else []

    if json_paths:
        for path in json_paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows[path.stem] = {
                dimension: int(score)
                for dimension, score in data.get("scores", {}).items()
                if score is not None
            }
        return rows

    if not outputs_zip.is_file():
        return rows

    with zipfile.ZipFile(outputs_zip) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".json"):
                continue
            paper_id = Path(name).stem
            with archive.open(name) as handle:
                data = json.loads(handle.read().decode("utf-8"))
            rows[paper_id] = {
                dimension: int(score)
                for dimension, score in data.get("scores", {}).items()
                if score is not None
            }

    return rows


VALUES = {
    "ARA": {
        "Rescience C": {"ACC": "", "F1": "", "Distance": "", "Abs Distance": ""},
        "GoldStandardDB": {"ACC": "", "F1": ""},
        "ReproBench": {"ACC": "", "F1": "", "Distance": "", "Abs Distance": ""},
    },
    "ReproScreener": {
        "Rescience C": {"ACC": "-", "F1": "-", "Distance": "-", "Abs Distance": "-"},
        "GoldStandardDB": {"ACC": "", "F1": ""},
        "ReproBench": {"ACC": "-", "F1": "-", "Distance": "-", "Abs Distance": "-"},
    },
    "ReplicatorAgent": {
        "Rescience C": {"ACC": "-", "F1": "-", "Distance": "-", "Abs Distance": "-"},
        "GoldStandardDB": {"ACC": "-", "F1": "-"},
        "ReproBench": {
            "ACC": "36.84",
            "F1": "36.67",
            "Distance": "--",
            "Abs Distance": "0.63",
        },  # GPT-5-mini
    },
}

DIRECT_VALUES = {
    row: {
        dataset: {"ACC": "-", "F1": "-"}
        for dataset in DATASETS
    }
    for row in ROWS
}

# ReplicatorAgent ReproBench values are already direct table values.
DIRECT_VALUES["ReplicatorAgent"]["ReproBench"]["ACC"] = "36.84"
DIRECT_VALUES["ReplicatorAgent"]["ReproBench"]["F1"] = "36.67"


######### CALCULATE PERFORMANCE OF ARA

### ON RESCIENCE C
rescience_metric_info = ROOT / "data" / "resciencec" / "metric_information"
rescience_truth_dir = rescience_metric_info / "outputs"
rescience_truth_zip = rescience_metric_info / "outputs.zip"
ara_rescience_csv = (
    ROOT
    / "data"
    / "experiments"
    / "batch_runs"
    / "rescience_c_gemini3_1_pro_T0_result.csv"
)

rescience_truth_by_paper = load_rescience_truth_outputs(
    rescience_truth_dir,
    rescience_truth_zip,
)

with ara_rescience_csv.open(newline="", encoding="utf-8") as f:
    ara_rescience_rows = list(csv.DictReader(f))

ara_rescience_true_values = []
ara_rescience_pred_values = []

for row in ara_rescience_rows:
    paperid = row["paperid"]
    if paperid not in rescience_truth_by_paper or "Overall" not in rescience_truth_by_paper[paperid]:
        continue

    ara_rescience_true_values.append(rescience_truth_by_paper[paperid]["Overall"])
    ara_rescience_pred_values.append(to_ordinal_class(float(row["R_0_4"])))

ara_rescience_accuracies, ara_rescience_f1_scores = macro_classification_scores(
    ara_rescience_true_values,
    ara_rescience_pred_values,
)
VALUES["ARA"]["Rescience C"]["ACC"] = mean_std(ara_rescience_accuracies)
VALUES["ARA"]["Rescience C"]["F1"] = mean_std(ara_rescience_f1_scores)

direct_acc, direct_f1 = multiclass_scores(
    ara_rescience_true_values,
    ara_rescience_pred_values,
)
DIRECT_VALUES["ARA"]["Rescience C"]["ACC"] = fmt_percent(direct_acc)
DIRECT_VALUES["ARA"]["Rescience C"]["F1"] = fmt_percent(direct_f1)

ara_rescience_distances = [
    pred - true for true, pred in zip(ara_rescience_true_values, ara_rescience_pred_values)
]
VALUES["ARA"]["Rescience C"]["Distance"] = mean_std_raw(ara_rescience_distances)
VALUES["ARA"]["Rescience C"]["Abs Distance"] = mean_std_raw(
    [abs(distance) for distance in ara_rescience_distances]
)

ara_rescience_rows_by_paper = {row["paperid"]: row for row in ara_rescience_rows}
RESCIENCE_DETAIL_VALUES = {}

for dimension, ara_column in RESCIENCE_DIMENSIONS:
    true_values = []
    pred_values = []

    for paperid, truth_scores in rescience_truth_by_paper.items():
        ara_row = ara_rescience_rows_by_paper.get(paperid)
        if ara_row is None or dimension not in truth_scores:
            continue
        if ara_row[ara_column] == "":
            continue

        true_values.append(truth_scores[dimension])
        if ara_column == "R_0_4":
            pred_values.append(to_ordinal_class(float(ara_row[ara_column])))
        else:
            pred_values.append(unit_to_ordinal_class(float(ara_row[ara_column])))

    accuracies_detail, f1_scores_detail = macro_classification_scores(true_values, pred_values)
    distances = [pred - true for true, pred in zip(true_values, pred_values)]
    abs_distances = [abs(distance) for distance in distances]

    RESCIENCE_DETAIL_VALUES[dimension] = {
        "ACC": mean_std(accuracies_detail),
        "F1": mean_std(f1_scores_detail),
        "Distance": mean_std_raw(distances),
        "Abs Distance": mean_std_raw(abs_distances),
    }


### ON GOLDSTANDARDDB
reproscreener_csv = ROOT / "data" / "reproscreener" / "reproducibility_scores.csv"
ara_reproscreener_base = ROOT / "data" / "reproscreener" / "ara_customized"
ara_reproscreener_outputs = ara_reproscreener_base / "outputs"
ara_reproscreener_outputs_zip = ara_reproscreener_base / "outputs.zip"

with reproscreener_csv.open(newline="", encoding="utf-8") as f:
    reproscreener_rows = list(csv.DictReader(f))

category_names = [
    column.removeprefix("manual_")
    for column in reproscreener_rows[0]
    if column.startswith("manual_")
]

ara_reproscreener_by_paper = load_ara_reproscreener_outputs(
    ara_reproscreener_outputs,
    ara_reproscreener_outputs_zip,
)

ara_gold_accuracies = []
ara_gold_f1_scores = []
ara_gold_true_all = []
ara_gold_pred_all = []

for category in category_names:
    true_values = []
    pred_values = []

    for row in reproscreener_rows:
        paper_id = row["paper_id"]
        categories = ara_reproscreener_by_paper.get(paper_id)
        if not categories or category not in categories:
            continue

        true_values.append(int(row[f"manual_{category}"]))
        pred_values.append(int(categories[category]["assessment"]))

    if not true_values:
        continue

    accuracy, f1 = binary_scores(true_values, pred_values)
    ara_gold_accuracies.append(accuracy)
    ara_gold_f1_scores.append(f1)

    ara_gold_true_all.extend(true_values)
    ara_gold_pred_all.extend(pred_values)

VALUES["ARA"]["GoldStandardDB"]["ACC"] = mean_std(ara_gold_accuracies)
VALUES["ARA"]["GoldStandardDB"]["F1"] = mean_std(ara_gold_f1_scores)

direct_acc, direct_f1 = binary_scores(ara_gold_true_all, ara_gold_pred_all)
DIRECT_VALUES["ARA"]["GoldStandardDB"]["ACC"] = fmt_percent(direct_acc)
DIRECT_VALUES["ARA"]["GoldStandardDB"]["F1"] = fmt_percent(direct_f1)


### ON REPROBENCH
reprobench_truth_csv = ROOT / "data" / "reprobench" / "reproducibility_scores.csv"
ara_reprobench_csv = (
    ROOT
    / "data"
    / "experiments"
    / "batch_runs"
    / "reprobench_gemini3_1_pro_T0_result.csv"
)

with reprobench_truth_csv.open(newline="", encoding="utf-8") as f:
    reprobench_truth_rows = list(csv.DictReader(f))

with ara_reprobench_csv.open(newline="", encoding="utf-8") as f:
    ara_reprobench_rows = list(csv.DictReader(f))

ground_truth_by_paper = {
    f"paper_{row['paper_id']}": int(row["ground_truth_reproducibility_score"])
    for row in reprobench_truth_rows
}

ara_true_values = []
ara_pred_values = []

for row in ara_reprobench_rows:
    paperid = row["paperid"]
    if paperid not in ground_truth_by_paper:
        continue

    ara_true_values.append(ground_truth_by_paper[paperid])
    ara_pred_values.append(to_ordinal_class(float(row["R_0_4"])))

ara_accuracies, ara_f1_scores = macro_classification_scores(
    ara_true_values,
    ara_pred_values,
)
VALUES["ARA"]["ReproBench"]["ACC"] = mean_std(ara_accuracies)
VALUES["ARA"]["ReproBench"]["F1"] = mean_std(ara_f1_scores)

direct_acc, direct_f1 = multiclass_scores(ara_true_values, ara_pred_values)
DIRECT_VALUES["ARA"]["ReproBench"]["ACC"] = fmt_percent(direct_acc)
DIRECT_VALUES["ARA"]["ReproBench"]["F1"] = fmt_percent(direct_f1)

ara_reprobench_distances = [
    pred - true for true, pred in zip(ara_true_values, ara_pred_values)
]
VALUES["ARA"]["ReproBench"]["Distance"] = mean_std_raw(ara_reprobench_distances)
VALUES["ARA"]["ReproBench"]["Abs Distance"] = mean_std_raw(
    [abs(distance) for distance in ara_reprobench_distances]
)


######### CALCULATE PERFORMANCE OF REPROSCREENER
accuracies = []
f1_scores = []
reproscreener_gold_true_all = []
reproscreener_gold_pred_all = []

for category in category_names:
    true_values = [int(row[f"manual_{category}"]) for row in reproscreener_rows]
    pred_values = [int(row[f"reproscreener_{category}"]) for row in reproscreener_rows]

    accuracy, f1 = binary_scores(true_values, pred_values)
    accuracies.append(accuracy)
    f1_scores.append(f1)

    reproscreener_gold_true_all.extend(true_values)
    reproscreener_gold_pred_all.extend(pred_values)

VALUES["ReproScreener"]["GoldStandardDB"]["ACC"] = mean_std(accuracies)
VALUES["ReproScreener"]["GoldStandardDB"]["F1"] = mean_std(f1_scores)

direct_acc, direct_f1 = binary_scores(
    reproscreener_gold_true_all,
    reproscreener_gold_pred_all,
)
DIRECT_VALUES["ReproScreener"]["GoldStandardDB"]["ACC"] = fmt_percent(direct_acc)
DIRECT_VALUES["ReproScreener"]["GoldStandardDB"]["F1"] = fmt_percent(direct_f1)


######### PRINT ORIGINAL TABLE
print("| dataset | metric | " + " | ".join(ROWS) + " |")
print("| --- | --- | " + " | ".join(["---"] * len(ROWS)) + " |")
for dataset in DATASETS:
    for metric in DATASET_METRICS[dataset]:
        cells = [dataset, metric]
        for row in ROWS:
            cells.append(VALUES[row][dataset][metric])
        print("| " + " | ".join(cells) + " |")


######### PRINT DIRECT POOLED TABLE
print()
print("Direct pooled ACC/F1")
print()
print("| dataset | metric | " + " | ".join(ROWS) + " |")
print("| --- | --- | " + " | ".join(["---"] * len(ROWS)) + " |")
for dataset in DATASETS:
    for metric in ("ACC", "F1"):
        cells = [dataset, metric]
        for row in ROWS:
            cells.append(DIRECT_VALUES[row][dataset][metric])
        print("| " + " | ".join(cells) + " |")


######### PRINT RESCIENCE DETAIL TABLE
print()
print("| ReScience C ARA score | ACC | F1 | Distance | Abs Distance |")
print("| --- | --- | --- | --- | --- |")
for dimension, _ in RESCIENCE_DIMENSIONS:
    values = RESCIENCE_DETAIL_VALUES[dimension]
    print(
        f"| {dimension} | {values['ACC']} | {values['F1']} | "
        f"{values['Distance']} | {values['Abs Distance']} |"
    )