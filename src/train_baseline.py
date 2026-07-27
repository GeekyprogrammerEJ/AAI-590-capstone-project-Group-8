"""
Train the from-scratch Transformer answer-selection baseline (Module 5).

Pipeline:
  1. Load the cleaned MCQ corpus (MedMCQA + MedQA) from Stage 1.
  2. Build a word-level vocabulary from the training split.
  3. Encode each item as four "[CLS] question [SEP] option" sequences.
  4. Train the from-scratch model with cross-entropy over the four options.
  5. Log the loss / validation-accuracy curve and save a checkpoint.

The goal for this module is to demonstrate that training has *successfully begun*
(a decreasing loss and above-chance validation accuracy), not to fully optimise
the model; optimisation is the focus of Module 6.

Run with:  python -m src.train_baseline
"""
import json
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from .baseline_model import MCQAnswerSelector

# --------------------------------------------------------------------------- #
# Reproducibility + device
# --------------------------------------------------------------------------- #
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------- #
# Hyperparameters (kept in one place for the write-up)
# --------------------------------------------------------------------------- #
HP = dict(
    n_train=30000, n_val=3000,   # subset sizes for a fast, demonstrative run
    vocab_size=20000, max_q=64, max_opt=24, max_len=96,
    dim=160, n_heads=4, n_layers=2, ff_dim=320, dropout=0.1,
    batch_size=32, epochs=4, lr=5e-4, weight_decay=0.01, warmup_frac=0.05,
)

PAD, UNK, CLS, SEP = "[PAD]", "[UNK]", "[CLS]", "[SEP]"
_TOKEN = re.compile(r"\b\w[\w'-]*\b")


def tokenize(text: str):
    return _TOKEN.findall(str(text).lower())


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
def build_vocab(texts, max_size):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    specials = [PAD, UNK, CLS, SEP]
    most = [w for w, _ in counter.most_common(max_size - len(specials))]
    itos = specials + most
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi


def encode_pair(question, option, stoi, max_q, max_opt, max_len):
    """Build one '[CLS] question [SEP] option' id sequence, padded to max_len."""
    unk = stoi[UNK]
    q = [stoi.get(w, unk) for w in tokenize(question)][:max_q]
    o = [stoi.get(w, unk) for w in tokenize(option)][:max_opt]
    ids = [stoi[CLS]] + q + [stoi[SEP]] + o
    ids = ids[:max_len]
    ids = ids + [stoi[PAD]] * (max_len - len(ids))
    return ids


def build_tensors(df, stoi, hp):
    """Return (ids tensor [N,4,L], label tensor [N]) for a dataframe of MCQ rows."""
    opt_cols = ["option_a", "option_b", "option_c", "option_d"]
    key_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    X, y = [], []
    for _, r in df.iterrows():
        seqs = [encode_pair(r["question"], r[c], stoi, hp["max_q"], hp["max_opt"], hp["max_len"])
                for c in opt_cols]
        X.append(seqs)
        y.append(key_to_idx[r["answer_key"]])
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for ids, labels in loader:
        ids, labels = ids.to(DEVICE), labels.to(DEVICE)
        pred = model(ids).argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def main():
    print(f"Device: {DEVICE}")
    df = pd.read_parquet(C.PROCESSED_DIR / "unified_medqa.parquet")
    mcq = df[df.source.isin(["medmcqa", "medqa"])].copy()

    # Deterministic shuffle, then carve out train / validation subsets.
    mcq = mcq.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    train_df = mcq.iloc[:HP["n_train"]]
    val_df = mcq.iloc[HP["n_train"]:HP["n_train"] + HP["n_val"]]
    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}")

    # Vocabulary is built ONLY on training text to avoid leakage.
    vocab_texts = pd.concat([train_df["question"], train_df["option_a"], train_df["option_b"],
                             train_df["option_c"], train_df["option_d"]])
    stoi = build_vocab(vocab_texts, HP["vocab_size"])
    print(f"Vocabulary: {len(stoi):,} tokens")

    Xtr, ytr = build_tensors(train_df, stoi, HP)
    Xva, yva = build_tensors(val_df, stoi, HP)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=HP["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=64)

    model = MCQAnswerSelector(
        vocab_size=len(stoi), pad_id=stoi[PAD], dim=HP["dim"], n_heads=HP["n_heads"],
        n_layers=HP["n_layers"], ff_dim=HP["ff_dim"], max_len=HP["max_len"], dropout=HP["dropout"],
    ).to(DEVICE)
    print(f"Trainable parameters: {model.num_parameters():,}")

    criterion = nn.CrossEntropyLoss()                 # categorical loss over the 4 options
    optimizer = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    total_steps = HP["epochs"] * len(train_loader)
    warmup = int(HP["warmup_frac"] * total_steps)

    def lr_at(step):                                  # linear warmup then linear decay
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    history = {"step": [], "train_loss": [], "val_acc": [], "val_step": []}
    chance = 1.0 / model.n_options
    print(f"Random-chance accuracy: {chance:.1%}\nStarting training...")

    step = 0
    t0 = time.time()
    for epoch in range(HP["epochs"]):
        model.train()
        running, since_log = 0.0, 0        # accumulate loss and count steps since last log
        for ids, labels in train_loader:
            ids, labels = ids.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(ids)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # stabilise early steps
            optimizer.step()
            scheduler.step()
            running += loss.item()
            since_log += 1
            step += 1
            if step % 50 == 0:
                avg = running / since_log   # divide by actual steps to avoid boundary artifacts
                running, since_log = 0.0, 0
                history["step"].append(step)
                history["train_loss"].append(avg)
                print(f"  epoch {epoch+1} step {step:4d}/{total_steps}  loss {avg:.4f}")
        acc = evaluate(model, val_loader)
        history["val_step"].append(step)
        history["val_acc"].append(acc)
        print(f"  >> epoch {epoch+1} validation accuracy: {acc:.3f} (chance {chance:.3f})")

    elapsed = time.time() - t0
    print(f"Training finished in {elapsed:.1f}s")

    # ---- Save artifacts -----------------------------------------------------
    (C.ROOT / "models").mkdir(exist_ok=True)
    torch.save({"model_state": model.state_dict(), "stoi": stoi, "hp": HP},
               C.ROOT / "models" / "baseline_mcq.pt")

    summary = {"device": str(DEVICE), "hyperparameters": HP,
               "trainable_parameters": model.num_parameters(),
               "chance_accuracy": chance, "final_val_accuracy": history["val_acc"][-1],
               "first_logged_loss": history["train_loss"][0] if history["train_loss"] else None,
               "last_logged_loss": history["train_loss"][-1] if history["train_loss"] else None,
               "train_seconds": round(elapsed, 1), "history": history}
    with open(C.REPORT_DIR / "baseline_training_log.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Training curve figure ---------------------------------------------
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.plot(history["step"], history["train_loss"], color=C.PALETTE["medmcqa"],
             linewidth=2, label="Training loss")
    ax1.set_xlabel("Training step"); ax1.set_ylabel("Cross-entropy loss", color=C.PALETTE["medmcqa"])
    ax1.tick_params(axis="y", labelcolor=C.PALETTE["medmcqa"])
    ax1.grid(True, color="#E6E6E6")
    ax2 = ax1.twinx()
    ax2.plot(history["val_step"], [a * 100 for a in history["val_acc"]], color=C.PALETTE["medqa"],
             marker="o", linewidth=2, label="Validation accuracy")
    ax2.axhline(chance * 100, ls="--", color="#888888", linewidth=1)
    ax2.set_ylabel("Validation accuracy (%)", color=C.PALETTE["medqa"])
    ax2.tick_params(axis="y", labelcolor=C.PALETTE["medqa"])
    ax2.annotate("random chance (25%)", (history["step"][0] if history["step"] else 0, chance*100+1),
                 fontsize=8, color="#666666")
    ax1.set_title("Figure 9  From-scratch baseline: training loss and validation accuracy",
                  fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig09_baseline_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved: models/baseline_mcq.pt, reports/baseline_training_log.json, "
          "figures/fig09_baseline_training.png")


if __name__ == "__main__":
    main()
