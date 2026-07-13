"""
Central configuration for the Capstone medical-QA data pipeline.

Holds filesystem paths, the dataset identifiers we pull from the Hugging Face
Hub, and a single colour-blind-safe plotting palette so that every figure in the
exploratory data analysis reads as one visual system.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
# All paths are resolved relative to the repository root (the parent of /src),
# so the pipeline runs the same way regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "reports"

for _d in (PROCESSED_DIR, FIG_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Source datasets (Hugging Face Hub identifiers)
# --------------------------------------------------------------------------- #
# Each entry: hub path, optional config name, and the splits we consume.
DATASETS = {
    "medmcqa":  {"path": "openlifescienceai/medmcqa",        "config": None},
    "medqa":    {"path": "GBaker/MedQA-USMLE-4-options",     "config": None},
    "pubmedqa": {"path": "qiaojin/PubMedQA",                 "config": "pqa_labeled"},
}

# --------------------------------------------------------------------------- #
# Plotting palette
# --------------------------------------------------------------------------- #
# Okabe-Ito: an eight-colour qualitative palette engineered to stay separable
# under the common forms of colour-vision deficiency. Hues are assigned to
# datasets in a FIXED order (never cycled), so a colour always means the same
# source across every figure.
PALETTE = {
    "medmcqa":  "#0072B2",  # blue
    "medqa":    "#D55E00",  # vermillion
    "pubmedqa": "#009E73",  # bluish green
}
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]

# Single hue used for magnitude (sequential) plots; diverging map for the
# correlation heatmap uses a neutral-grey midpoint at zero.
SEQUENTIAL_CMAP = "Blues"
DIVERGING_CMAP = "RdBu_r"

# Human-readable names for captions and the written report.
PRETTY_NAME = {
    "medmcqa": "MedMCQA",
    "medqa": "MedQA (USMLE)",
    "pubmedqa": "PubMedQA",
}
