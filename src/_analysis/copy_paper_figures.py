"""Copy publication body figures to figures/paper/.

Body figures (per analysis_summary.md):
  Fig 1  — fig1_model_comparison            (Ch. 1 — model comparison)
  Fig 3  — gemini3_1_pro_T0_structural_vs_content  (Ch. 2 — content bottleneck)
  Fig 4  — gemini3_1_pro_T0_layer_content   (Ch. 2 — source-layer gap)
  Fig 7  — gemini3_1_pro_T0_metadata_richness      (Ch. 2 — repo availability)
  Fig 8  — fig8_cross_validation            (Ch. 1+2 — cross-notebook validation)
"""

import sys
import shutil
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR  = Path("figures")
DEST_DIR = Path("figures/paper")
DEST_DIR.mkdir(parents=True, exist_ok=True)

PAPER_FIGURES = [
    "fig1_model_comparison",
    "gemini3_1_pro_T0_structural_vs_content",
    "gemini3_1_pro_T0_layer_content",
    "gemini3_1_pro_T0_metadata_richness",
    "fig8_cross_validation",
    "gemini3_1_pro_T0_wiring_metrics",
    "gemini3_1_pro_T0_node_counts_vs_repro",
    "gemini3_1_pro_T0_repro_by_year",
    "gemini3_1_pro_T0_paper_ranking",
]

for stem in PAPER_FIGURES:
    for ext in (".pdf", ".png"):
        src = SRC_DIR / f"{stem}{ext}"
        dst = DEST_DIR / f"{stem}{ext}"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  copied  {src.name}  →  {dst}")
        else:
            print(f"  MISSING {src}")

print(f"\nDone. {len(list(DEST_DIR.iterdir()))} files in {DEST_DIR.resolve()}")
