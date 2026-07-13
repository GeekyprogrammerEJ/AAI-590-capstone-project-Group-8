"""
Stage 1 - Data loading, cleaning, and unification.

This module pulls the three source medical question-answering datasets from the
Hugging Face Hub, applies a consistent set of cleaning steps, folds them into a
single unified schema, derives numeric features for the exploratory analysis, and
writes:

    data/processed/unified_medqa.parquet   - the cleaned, unified corpus
    reports/cleaning_report.json            - row counts at every cleaning step

Run directly with:  python -m src.prepare_data
"""
import json
import warnings

import pandas as pd
from datasets import load_dataset

from . import config as C
from .text_utils import clean_text, word_count, approx_tokens

warnings.filterwarnings("ignore")

# The MedMCQA "correct option pointer" is a ClassLabel index; map it to a letter.
COP_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


# --------------------------------------------------------------------------- #
# Raw loaders - one per source. Each returns a *labelled* pandas DataFrame in a
# common set of columns before any cleaning is applied.
# --------------------------------------------------------------------------- #
def load_medmcqa() -> pd.DataFrame:
    """MedMCQA: large 4-option MCQ set with subject/topic metadata.

    The public test split is unlabelled (no gold answer), so we build the EDA
    corpus from the train and validation splits, which carry gold labels.
    """
    ds = load_dataset(C.DATASETS["medmcqa"]["path"])
    frames = []
    for split in ("train", "validation"):
        df = ds[split].to_pandas()
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    out = pd.DataFrame({
        "source": "medmcqa",
        "split": df["split"],
        "question": df["question"],
        "option_a": df["opa"], "option_b": df["opb"],
        "option_c": df["opc"], "option_d": df["opd"],
        "answer_key": df["cop"].map(COP_TO_LETTER),
        "subject": df["subject_name"],
        # choice_type distinguishes single-answer from (rare) multi-answer items.
        "choice_type": df["choice_type"],
        "context": "",
    })
    return out


def load_medqa() -> pd.DataFrame:
    """MedQA (USMLE): 4-option board-exam MCQs; options arrive as a dict A-D."""
    ds = load_dataset(C.DATASETS["medqa"]["path"])
    frames = []
    for split in ("train", "test"):
        df = ds[split].to_pandas()
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    opts = df["options"].apply(lambda d: dict(d) if d is not None else {})
    out = pd.DataFrame({
        "source": "medqa",
        "split": df["split"],
        "question": df["question"],
        "option_a": opts.apply(lambda d: d.get("A", "")),
        "option_b": opts.apply(lambda d: d.get("B", "")),
        "option_c": opts.apply(lambda d: d.get("C", "")),
        "option_d": opts.apply(lambda d: d.get("D", "")),
        "answer_key": df["answer_idx"],
        # meta_info records the USMLE step; we expose it as the "subject" facet.
        "subject": df["meta_info"],
        "choice_type": "single",
        "context": "",
    })
    return out


def load_pubmedqa() -> pd.DataFrame:
    """PubMedQA: yes/no/maybe questions answered against a research abstract.

    This is not a multiple-choice task, so the option columns are left empty and
    the abstract is stored in the shared ``context`` column.
    """
    ds = load_dataset(C.DATASETS["pubmedqa"]["path"], C.DATASETS["pubmedqa"]["config"])
    df = ds["train"].to_pandas()

    def join_context(ctx):
        # ``context`` is a struct with a list of abstract paragraphs.
        if isinstance(ctx, dict) and "contexts" in ctx and ctx["contexts"] is not None:
            return " ".join(list(ctx["contexts"]))
        return ""

    out = pd.DataFrame({
        "source": "pubmedqa",
        "split": "labeled",
        "question": df["question"],
        "option_a": "", "option_b": "", "option_c": "", "option_d": "",
        "answer_key": df["final_decision"],      # yes / no / maybe
        "subject": "biomedical_research",
        "choice_type": "yes_no_maybe",
        "context": df["context"].apply(join_context),
    })
    return out


# --------------------------------------------------------------------------- #
# Cleaning + unification
# --------------------------------------------------------------------------- #
TEXT_COLS = ["question", "option_a", "option_b", "option_c", "option_d", "context"]


def clean_and_unify(raw: pd.DataFrame, log: dict) -> pd.DataFrame:
    """Apply the shared cleaning steps and derive numeric features.

    Every step appends a row-count checkpoint to ``log`` so the written report can
    state exactly how many records each operation removed.
    """
    df = raw.copy()
    log["01_loaded_total"] = len(df)

    # -- Step 1: normalise all free-text fields (unicode, HTML, whitespace). -----
    for col in TEXT_COLS:
        df[col] = df[col].apply(clean_text)

    # -- Step 2: drop records with no question text. ----------------------------
    before = len(df)
    df = df[df["question"].str.len() > 0]
    log["02_dropped_empty_question"] = before - len(df)

    # -- Step 3: drop records missing a gold answer key. ------------------------
    before = len(df)
    df = df[df["answer_key"].notna() & (df["answer_key"].astype(str).str.len() > 0)]
    log["03_dropped_missing_answer"] = before - len(df)

    # -- Step 4: MCQ validity - a 4-option item must have >=2 non-empty options
    #            and its keyed option must not be blank. PubMedQA is exempt. ------
    def mcq_valid(row):
        if row["source"] == "pubmedqa":
            return True
        opts = [row["option_a"], row["option_b"], row["option_c"], row["option_d"]]
        non_empty = sum(1 for o in opts if o)
        keyed = {"A": 0, "B": 1, "C": 2, "D": 3}.get(row["answer_key"], None)
        keyed_ok = keyed is not None and keyed < 4 and bool(opts[keyed])
        return non_empty >= 2 and keyed_ok

    before = len(df)
    df = df[df.apply(mcq_valid, axis=1)]
    log["04_dropped_invalid_mcq"] = before - len(df)

    # -- Step 5: remove exact duplicate questions within a source. --------------
    # NOTE: MedMCQA's ``choice_type`` of "single"/"multi" denotes single- vs
    # multi-*sentence* reasoning, not multiple correct answers - every item still
    # has exactly one gold option - so we keep both and retain the flag as a
    # descriptive feature rather than dropping any rows here.
    before = len(df)
    df = df.drop_duplicates(subset=["source", "question", "answer_key"])
    log["05_dropped_duplicates"] = before - len(df)

    df = df.reset_index(drop=True)
    log["06_clean_total"] = len(df)

    # -- Step 7: derive numeric features used throughout the EDA. ----------------
    df["q_word_len"] = df["question"].apply(word_count)
    df["q_char_len"] = df["question"].str.len()
    opt_cols = ["option_a", "option_b", "option_c", "option_d"]
    df["opt_word_len_mean"] = df[opt_cols].applymap(word_count).mean(axis=1)
    df["n_options"] = df[opt_cols].apply(lambda r: sum(1 for o in r if o), axis=1)
    df["context_word_len"] = df["context"].apply(word_count)
    # Approximate full-prompt token load (question + options + context).
    df["approx_prompt_tokens"] = (
        df["question"].apply(approx_tokens)
        + df[opt_cols].applymap(approx_tokens).sum(axis=1)
        + df["context"].apply(approx_tokens)
    )
    # Numeric index of the correct option, used to check for positional bias.
    df["answer_idx_num"] = df["answer_key"].map({"A": 0, "B": 1, "C": 2, "D": 3})
    return df


def build(save: bool = True) -> pd.DataFrame:
    """End-to-end Stage 1: load all sources, clean, unify, and persist."""
    log = {}
    raw = pd.concat([load_medmcqa(), load_medqa(), load_pubmedqa()], ignore_index=True)
    unified = clean_and_unify(raw, log)

    # Per-source clean counts for the report.
    log["per_source_clean_counts"] = (
        unified["source"].value_counts().to_dict()
    )

    if save:
        out_path = C.PROCESSED_DIR / "unified_medqa.parquet"
        unified.to_parquet(out_path, index=False)
        with open(C.REPORT_DIR / "cleaning_report.json", "w") as f:
            json.dump(log, f, indent=2)
        print(f"Saved {len(unified):,} cleaned rows -> {out_path}")
        print("Cleaning checkpoints:")
        for k, v in log.items():
            print(f"  {k}: {v}")
    return unified


if __name__ == "__main__":
    build()
