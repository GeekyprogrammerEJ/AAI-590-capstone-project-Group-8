# Results & Optimization Summary

Measured on a held-out set of 500 medical multiple-choice questions
(exact-match accuracy; random chance = 25%). Produced by
`notebooks/qlora_finetuning_colab.ipynb` on an NVIDIA A100-SXM4-40GB.

## Headline comparison

| Model | Trainable params | Accuracy |
|-------|------------------|----------|
| Random chance | — | 25.0% |
| From-scratch Transformer baseline | 3.6 M | 29.1% |
| Base 7B LLM (Mistral 7B, zero-shot) | 0 (frozen) | 55.4% |
| **QLoRA-fine-tuned 7B LLM** | 41.9 M (0.58%) | **60.8%** |

- Pretraining accounts for most of the competence: 29.1% → 55.4% (**+26.3**).
- QLoRA fine-tuning adds a further **+5.4** points (55.4% → 60.8%), a ~12%
  relative reduction in error.
- The fine-tuned model trains only **0.58%** of the base model's 7.28 B
  parameters (41,943,040 of 7,283,675,136).

## LoRA-rank sweep (optimization)

Controlled sweep on a fixed 4,000-example subset, one epoch each, alpha = 2 × rank:

| LoRA rank | Alpha | Val. accuracy |
|-----------|-------|---------------|
| 8 | 16 | 59.2% |
| **16** | **32** | **60.0%** (best) |
| 32 | 64 | 57.6% |

- Rank 16 is the capacity sweet spot; rank 32 slightly overfits, rank 8 slightly
  underfits. The full 8,000-example run at rank 16 reached 60.8%.

## Configuration (reported run)

- Base model: `mistralai/Mistral-7B-Instruct-v0.2`, 4-bit NF4 (double quant)
- LoRA: r = 16, alpha = 32, dropout = 0.05, all attention + MLP projections
- Optimizer: paged AdamW 8-bit, lr = 2e-4, cosine schedule, warmup 3%
- 1 epoch, effective batch 16 (batch 2 × grad-accum 8), max seq len 768
- Training: ~47 min on one A100; 8,000 examples

*Numbers here are reproduced by the executed notebook in `notebooks/`.*
