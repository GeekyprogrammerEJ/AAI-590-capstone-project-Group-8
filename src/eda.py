"""
Stage 2 - Exploratory Data Analysis.

Reads the cleaned, unified corpus produced by ``prepare_data`` and generates the
figures and summary statistics used in the Data Summary section of the report:

    figures/fig01..fig08_*.png     - publication-style EDA figures
    reports/eda_stats.json          - machine-readable summary statistics

Design notes: every figure uses the shared Okabe-Ito palette from ``config`` so a
colour always denotes the same dataset; magnitude plots use a single sequential
hue and the correlation heatmap uses a diverging map with a neutral midpoint at
zero, matching the polarity of a correlation coefficient.

Run directly with:  python -m src.eda
"""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")                       # headless / file-only rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .text_utils import content_words

# --------------------------------------------------------------------------- #
# Shared plotting style - applied once so all figures read as one system.
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
})

SOURCES = ["medmcqa", "medqa", "pubmedqa"]

# Stop-words removed before counting the most frequent clinical vocabulary.
# English function words plus a few multiple-choice boilerplate tokens.
STOPWORDS = set("""
a an the and or of to in for with on at by from is are was were be been being as
this that these those which who whom whose what when where why how it its their his
her they them he she you your we our not no yes all any both each few more most
other some such than too very can will just should now then here there about into
following except true false all-of-the-above none most likely cause caused causes
patient year old years age man woman male female present presents presenting history
has have had having seen see shows show shown showing used use uses using given give
after before during common right left past normal min hour hours day days week weeks
found feature features finding findings associated known called due likely best next
which one two three four also may might would could per among within without via
because but however although while if then than so out over under between only
""".split())


def load() -> pd.DataFrame:
    df = pd.read_parquet(C.PROCESSED_DIR / "unified_medqa.parquet")
    return df


def _bar_labels(ax, fmt="{:,.0f}", pad=3, fontsize=9):
    """Write value labels above vertical bars (selective direct labelling)."""
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h) and h > 0:
            ax.annotate(fmt.format(h), (p.get_x() + p.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=fontsize, xytext=(0, pad),
                        textcoords="offset points", color="#333333")


def save(fig, name):
    path = C.FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path.name)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_dataset_sizes(df, stats):
    """Fig 1 - cleaned record counts per dataset, split by train/eval."""
    piv = (df.assign(kind=np.where(df["split"].isin(["train", "labeled"]), "train", "eval"))
             .groupby(["source", "kind"]).size().unstack(fill_value=0)
             .reindex(SOURCES))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bottom = np.zeros(len(piv))
    for kind, hatch in [("train", None), ("eval", "//")]:
        vals = piv.get(kind, pd.Series(0, index=piv.index)).values
        ax.bar([C.PRETTY_NAME[s] for s in piv.index], vals, bottom=bottom,
               color=[C.PALETTE[s] for s in piv.index], edgecolor="white",
               hatch=hatch, label=kind, alpha=0.95 if kind == "train" else 0.55)
        bottom += vals
    for i, s in enumerate(piv.index):
        total = piv.loc[s].sum()
        ax.annotate(f"{total:,}", (i, total), ha="center", va="bottom",
                    fontsize=10, fontweight="bold", xytext=(0, 3), textcoords="offset points")
    ax.set_ylabel("Cleaned questions")
    ax.set_title("Figure 1  Dataset sizes after cleaning (train vs. held-out)")
    ax.set_yscale("log")
    ax.set_ylim(300, 4e5)          # floor below PubMedQA's 1,000 so its bar is visible
    ax.legend(title="Split", frameon=False)
    save(fig, "fig01_dataset_sizes.png")
    stats["dataset_sizes"] = df["source"].value_counts().to_dict()


def fig_question_length(df, stats):
    """Fig 2 - distribution of question length (words) per dataset."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bins = np.linspace(0, 120, 61)
    for s in SOURCES:
        vals = df.loc[df.source == s, "q_word_len"].clip(upper=120)
        ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=2,
                color=C.PALETTE[s], label=C.PRETTY_NAME[s])
    ax.set_xlabel("Question length (words)")
    ax.set_ylabel("Density")
    ax.set_title("Figure 2  Question length distribution by dataset")
    ax.legend(frameon=False)
    save(fig, "fig02_question_length.png")
    stats["question_word_len"] = {
        s: {"median": float(df.loc[df.source == s, "q_word_len"].median()),
            "p90": float(df.loc[df.source == s, "q_word_len"].quantile(0.9)),
            "max": int(df.loc[df.source == s, "q_word_len"].max())}
        for s in SOURCES}


def fig_answer_balance(df, stats):
    """Fig 3 - positional balance of the correct option (A-D).

    A strong skew toward one letter would let a model exploit a positional
    shortcut instead of learning the medicine, so this is a key sanity check.
    """
    mcq = df[df.source.isin(["medmcqa", "medqa"])]
    order = ["A", "B", "C", "D"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    width = 0.38
    x = np.arange(len(order))
    for i, s in enumerate(["medmcqa", "medqa"]):
        counts = (mcq[mcq.source == s]["answer_key"].value_counts(normalize=True)
                  .reindex(order).fillna(0) * 100)
        ax.bar(x + (i - 0.5) * width, counts.values, width,
               color=C.PALETTE[s], label=C.PRETTY_NAME[s], edgecolor="white")
    ax.axhline(25, color="#888888", ls="--", lw=1)
    ax.annotate("uniform = 25%", (-0.45, 25.6), fontsize=8.5, color="#666666")
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylim(0, 33)
    ax.set_xlabel("Correct option position")
    ax.set_ylabel("Share of questions (%)")
    ax.set_title("Figure 3  Correct-answer position balance")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig03_answer_balance.png")
    stats["answer_key_balance_pct"] = {
        s: (mcq[mcq.source == s]["answer_key"].value_counts(normalize=True)
            .reindex(order).fillna(0).round(4) * 100).to_dict()
        for s in ["medmcqa", "medqa"]}


def fig_subjects(df, stats):
    """Fig 4 - MedMCQA subject distribution (magnitude -> single sequential hue)."""
    subj = df[df.source == "medmcqa"]["subject"].value_counts().head(12).iloc[::-1]
    norm = plt.Normalize(subj.min(), subj.max())
    colors = plt.get_cmap(C.SEQUENTIAL_CMAP)(norm(subj.values) * 0.7 + 0.3)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(subj.index, subj.values, color=colors, edgecolor="white")
    for i, v in enumerate(subj.values):
        ax.annotate(f"{v:,}", (v, i), va="center", fontsize=8.5,
                    xytext=(3, 0), textcoords="offset points", color="#333333")
    ax.set_xlabel("Questions")
    ax.set_title("Figure 4  MedMCQA questions by medical subject (top 12)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig04_medmcqa_subjects.png")
    stats["medmcqa_subject_counts"] = (
        df[df.source == "medmcqa"]["subject"].value_counts().to_dict())


def fig_pubmedqa_labels(df, stats):
    """Fig 5 - PubMedQA yes/no/maybe class balance."""
    lab = df[df.source == "pubmedqa"]["answer_key"].value_counts().reindex(
        ["yes", "no", "maybe"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = {"yes": "#009E73", "no": "#D55E00", "maybe": "#E69F00"}
    ax.bar(lab.index, lab.values, color=[cmap[k] for k in lab.index], edgecolor="white")
    _bar_labels(ax)
    ax.set_ylabel("Questions")
    ax.set_title("Figure 5  PubMedQA answer distribution")
    ax.grid(axis="x", visible=False)
    save(fig, "fig05_pubmedqa_labels.png")
    stats["pubmedqa_label_counts"] = lab.astype(int).to_dict()


def fig_prompt_tokens(df, stats):
    """Fig 6 - approximate full-prompt token load per dataset (compute/context)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    data = [df.loc[df.source == s, "approx_prompt_tokens"].clip(upper=800) for s in SOURCES]
    bp = ax.boxplot(data, vert=True, patch_artist=True, showfliers=False,
                    widths=0.55, medianprops=dict(color="#222222", linewidth=1.5))
    for patch, s in zip(bp["boxes"], SOURCES):
        patch.set_facecolor(C.PALETTE[s]); patch.set_alpha(0.75); patch.set_edgecolor("white")
    ax.set_xticklabels([C.PRETTY_NAME[s] for s in SOURCES])
    ax.set_ylabel("Approx. prompt tokens")
    ax.set_title("Figure 6  Estimated prompt length (question + options + context)")
    save(fig, "fig06_prompt_tokens.png")
    stats["approx_prompt_tokens"] = {
        s: {"median": float(df.loc[df.source == s, "approx_prompt_tokens"].median()),
            "p95": float(df.loc[df.source == s, "approx_prompt_tokens"].quantile(0.95))}
        for s in SOURCES}


def fig_corr(df, stats):
    """Fig 7 - correlation among derived numeric features (diverging, 0-centred)."""
    feats = ["q_word_len", "q_char_len", "opt_word_len_mean", "n_options",
             "context_word_len", "approx_prompt_tokens", "answer_idx_num"]
    labels = ["Q words", "Q chars", "Opt words", "# options",
              "Context words", "Prompt tokens", "Answer index"]
    # ``answer_idx_num`` is defined only for MCQ (which carry no context) and
    # ``context_word_len`` is non-zero only for PubMedQA (which has no A-D index),
    # so their pairwise correlation is undefined; fill such structural NaNs with 0.
    corr = df[feats].corr().fillna(0.0)
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(corr.values, cmap=C.DIVERGING_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(feats)):
        for j in range(len(feats)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.55 else "#222222")
    ax.set_title("Figure 7  Correlation of derived numeric features")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.grid(False)
    save(fig, "fig07_correlation.png")
    stats["feature_correlations"] = corr.round(3).to_dict()


def fig_top_terms(df, stats):
    """Fig 8 - most frequent clinical terms across all questions."""
    counter = Counter()
    # Sample for speed; the top of the distribution is stable under sampling.
    sample = df["question"].sample(min(40000, len(df)), random_state=42)
    for q in sample:
        counter.update(content_words(q, STOPWORDS))
    top = counter.most_common(20)[::-1]
    terms = [t for t, _ in top]
    counts = [c for _, c in top]
    norm = plt.Normalize(min(counts), max(counts))
    colors = plt.get_cmap(C.SEQUENTIAL_CMAP)(norm(counts) * 0.7 + 0.3)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(terms, counts, color=colors, edgecolor="white")
    ax.set_xlabel("Occurrences (40k-question sample)")
    ax.set_title("Figure 8  Most frequent clinical terms in questions")
    ax.grid(axis="y", visible=False)
    save(fig, "fig08_top_terms.png")
    stats["top_terms"] = dict(counter.most_common(25))


def main():
    df = load()
    print(f"Loaded {len(df):,} cleaned rows for EDA")
    stats = {"n_rows": int(len(df))}
    fig_dataset_sizes(df, stats)
    fig_question_length(df, stats)
    fig_answer_balance(df, stats)
    fig_subjects(df, stats)
    fig_pubmedqa_labels(df, stats)
    fig_prompt_tokens(df, stats)
    fig_corr(df, stats)
    fig_top_terms(df, stats)
    with open(C.REPORT_DIR / "eda_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved statistics -> reports/eda_stats.json")


if __name__ == "__main__":
    main()
