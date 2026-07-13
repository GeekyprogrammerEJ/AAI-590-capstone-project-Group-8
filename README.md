# Capstone: Fine-Tuning a Compact Open LLM for Medical Question Answering

AAI-590 Capstone Project — M.S. Applied Artificial Intelligence, University of San Diego.
Team: Evin Joy, Sabina George, Jagadeesh Kumar Sellapan.

This repository contains the data pipeline and exploratory data analysis (EDA)
for our medical question-answering (QA) capstone. It downloads three public
medical QA datasets, cleans and unifies them into a single schema, and generates
the statistics and figures used in the report's **Data Summary** section
(Assignment 3.1 / Module 3).

## What it does

| Stage | Module | Output |
|-------|--------|--------|
| **1 — Prepare** | `src/prepare_data.py` | `data/processed/unified_medqa.parquet`, `reports/cleaning_report.json` |
| **2 — EDA**     | `src/eda.py`          | `figures/fig01…fig08_*.png`, `reports/eda_stats.json` |

An executed, output-rich walkthrough of both stages is in
[`notebooks/data_cleaning_and_eda.ipynb`](notebooks/data_cleaning_and_eda.ipynb).

## Datasets (loaded from the Hugging Face Hub)

| Dataset | Source | Task | Cleaned size |
|---------|--------|------|--------------|
| MedMCQA | `openlifescienceai/medmcqa` | 4-option MCQ | 186,906 |
| MedQA (USMLE) | `GBaker/MedQA-USMLE-4-options` | 4-option MCQ | 11,451 |
| PubMedQA | `qiaojin/PubMedQA` (`pqa_labeled`) | Yes/No/Maybe | 1,000 |

## Repository layout

```
capstone-medqa/
├── src/
│   ├── config.py         # paths, dataset IDs, plotting palette
│   ├── text_utils.py     # text normalization + length/token helpers
│   ├── prepare_data.py   # Stage 1: load, clean, unify, derive features
│   └── eda.py            # Stage 2: summary statistics + figures
├── notebooks/
│   └── data_cleaning_and_eda.ipynb   # executed end-to-end walkthrough
├── data/processed/       # cleaned unified corpus (parquet)
├── figures/              # generated EDA figures (PNG)
├── reports/              # cleaning_report.json, eda_stats.json
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
python -m src.prepare_data      # Stage 1 — downloads + cleans (~a few minutes on first run)
python -m src.eda               # Stage 2 — writes figures/ and reports/
```

The raw datasets are cached by the `datasets` library (default: `~/.cache/huggingface`)
and are not committed to the repository.

## Cleaning steps (see `prepare_data.py`)

1. Normalize all text (unicode NFKC, HTML un-escape, whitespace collapse).
2. Drop records with an empty question or missing gold answer.
3. Drop structurally invalid multiple-choice items (blank keyed option / < 2 options).
4. Remove exact duplicate questions within a source.
5. Derive numeric features (question/option/context length, option count,
   approximate prompt tokens, correct-option index).

## EDA figures

Dataset sizes, question-length distributions, correct-answer position balance,
MedMCQA subject coverage, PubMedQA class balance, prompt-token budget, a
feature-correlation heatmap, and the most frequent clinical terms.

## Notes

- MedMCQA's `choice_type = single/multi` denotes single- vs multi-*sentence*
  reasoning (not multiple correct answers); all such items are kept.
- The public MedMCQA test split is unlabeled, so EDA uses its train + validation splits.
- A generative AI assistant was used to help scaffold and document this code; all
  analysis was reviewed and verified by the team.
