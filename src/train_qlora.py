"""
QLoRA fine-tuning of a compact open-weight LLM for medical question answering
(the project's *primary* model). Script form of
``notebooks/qlora_finetuning_colab.ipynb``.

QLoRA quantises the pretrained weights to 4-bit and freezes them, training only
small injected low-rank adapters; this makes a 7-8B model adaptable on a single
GPU. The from-scratch model in ``baseline_model.py`` is the comparison baseline.

REQUIRES A CUDA GPU (this will not run on CPU / Apple-MPS) and:
    pip install "bitsandbytes==0.43.3" "transformers==4.44.2" "peft==0.12.0" \\
                "accelerate==0.33.0" "trl==0.9.6" "datasets==2.21.0"

For gated base models (Mistral, Llama), accept the license on the Hugging Face
Hub and authenticate with `huggingface-cli login` or the HF_TOKEN env variable.

Run with:  python -m src.train_qlora
"""
import random
import re

import torch
from datasets import Dataset, load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CFG = dict(
    # A compact open-weight LLM. Mistral is gated on the Hub (accept its license
    # and authenticate), or switch to an ungated model such as
    # "Qwen/Qwen2.5-7B-Instruct".
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    n_train=8000, n_val=500, max_seq_len=768,
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    learning_rate=2e-4, epochs=1, batch_size=2, grad_accum=8,
    output_dir="qlora-medqa-adapter",
)

LETTERS = "ABCD"
PROMPT = (
    "You are a medical expert answering a multiple-choice question. "
    "Select the single best option.\n\n"
    "Question: {q}\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
    "Answer:"
)


# --------------------------------------------------------------------------- #
# Data loading and instruction formatting
# --------------------------------------------------------------------------- #
def load_records(n):
    """Return up to n unified {question, options, answer_idx} records from
    MedMCQA (topped up with MedQA if needed)."""
    out = []
    for r in load_dataset("openlifescienceai/medmcqa", split="train"):
        opts = [r["opa"], r["opb"], r["opc"], r["opd"]]
        if all(o for o in opts) and r["cop"] in (0, 1, 2, 3):
            out.append({"question": r["question"].strip(), "options": opts,
                        "answer_idx": int(r["cop"])})
        if len(out) >= n:
            return out
    for r in load_dataset("GBaker/MedQA-USMLE-4-options", split="train"):
        o = r["options"]
        out.append({"question": r["question"].strip(),
                    "options": [o["A"], o["B"], o["C"], o["D"]],
                    "answer_idx": "ABCD".index(r["answer_idx"])})
        if len(out) >= n:
            break
    return out


def format_example(rec, with_answer=True, shuffle=True):
    """Render one record as prompt text; optionally shuffle options (and remap the
    gold index) to neutralise positional bias. Returns (text, gold_index)."""
    opts, idx = list(rec["options"]), rec["answer_idx"]
    if shuffle:
        order = list(range(4))
        random.shuffle(order)
        opts = [rec["options"][i] for i in order]
        idx = order.index(rec["answer_idx"])
    text = PROMPT.format(q=rec["question"], a=opts[0], b=opts[1], c=opts[2], d=opts[3])
    if with_answer:
        text = f"{text} {LETTERS[idx]}. {opts[idx]}"
    return text, idx


def build_datasets(cfg, eos_token):
    """Return (train Dataset with a 'text' field, list of validation records)."""
    recs = load_records(cfg["n_train"] + cfg["n_val"])
    random.shuffle(recs)
    train_recs = recs[:cfg["n_train"]]
    val_recs = recs[cfg["n_train"]:cfg["n_train"] + cfg["n_val"]]
    # Training text ends with the answer plus EOS so the model learns to stop.
    texts = [format_example(r, with_answer=True)[0] + eos_token for r in train_recs]
    return Dataset.from_dict({"text": texts}), val_recs


# --------------------------------------------------------------------------- #
# Model (4-bit base + LoRA adapters)
# --------------------------------------------------------------------------- #
def build_model_and_tokenizer(cfg):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,   # torch.bfloat16 on A100/L4
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, tokenizer, val_recs):
    """Greedy-decode one answer letter per question; return accuracy."""
    model.eval()
    correct = 0
    for rec in val_recs:
        prompt, gold_idx = format_example(rec, with_answer=False, shuffle=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=4, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        m = re.search(r"[ABCD]", gen.upper())
        pred_idx = "ABCD".index(m.group(0)) if m else 0
        correct += int(pred_idx == gold_idx)
    acc = correct / max(len(val_recs), 1)
    print(f"Validation accuracy: {acc:.3f} on {len(val_recs)} questions (chance 0.25)")
    return acc


# --------------------------------------------------------------------------- #
# Training entry point
# --------------------------------------------------------------------------- #
def main():
    assert torch.cuda.is_available(), "A CUDA GPU is required for QLoRA training."
    random.seed(42)
    torch.manual_seed(42)

    model, tokenizer = build_model_and_tokenizer(CFG)
    train_ds, val_recs = build_datasets(CFG, tokenizer.eos_token)
    print(f"Train examples: {len(train_ds):,} | Val examples: {len(val_recs):,}")

    train_args = TrainingArguments(
        output_dir=CFG["output_dir"],
        per_device_train_batch_size=CFG["batch_size"],
        gradient_accumulation_steps=CFG["grad_accum"],
        learning_rate=CFG["learning_rate"],
        num_train_epochs=CFG["epochs"],
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,                       # bf16=True on A100/L4 instead
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        dataset_text_field="text",
        max_seq_length=CFG["max_seq_len"],
        args=train_args,
    )
    trainer.train()

    evaluate(model, tokenizer, val_recs)

    model.save_pretrained(CFG["output_dir"])
    tokenizer.save_pretrained(CFG["output_dir"])
    print("Saved adapter to:", CFG["output_dir"])


if __name__ == "__main__":
    main()
