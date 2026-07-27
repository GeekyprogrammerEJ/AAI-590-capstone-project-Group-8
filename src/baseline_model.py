"""
From-scratch Transformer answer-selection network (baseline model).

This module implements a compact Transformer encoder *from scratch* in PyTorch:
the multi-head self-attention, the encoder block, and the multiple-choice scoring
head are all written explicitly rather than relying on ``nn.Transformer`` or
``nn.MultiheadAttention``. It serves as the project's transparent, fully trainable
baseline against which the primary QLoRA-fine-tuned large language model is
compared.

Task framing: medical multiple-choice question answering as answer selection. The
same encoder reads the question paired with each candidate option, produces a
score per option, and a softmax over the four scores yields a distribution the
model is trained to match to the correct option with cross-entropy loss.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention, implemented from scratch."""

    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        assert dim % n_heads == 0, "embedding dim must divide evenly among heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        # Separate projections for queries, keys, values, and the output mix.
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        # x: (B, L, D); pad_mask: (B, L) with True at padding positions.
        B, L, D = x.shape
        # Project and split into heads -> (B, H, L, head_dim).
        def split(t):
            return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))

        # Attention scores and masking of padded keys.
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale      # (B, H, L, L)
        scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, v)                                     # (B, H, L, head_dim)

        # Recombine heads and apply the output projection.
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(ctx)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block: attention + position-wise feed-forward,
    each wrapped in a residual connection (Vaswani et al., 2017)."""

    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        x = x + self.dropout(self.attn(self.norm1(x), pad_mask))   # residual around attention
        x = x + self.dropout(self.ff(self.norm2(x)))               # residual around feed-forward
        return x


class TransformerEncoder(nn.Module):
    """Token + learned positional embeddings followed by a stack of encoder blocks.
    A prepended [CLS] position provides the pooled sequence representation."""

    def __init__(self, vocab_size, dim, n_heads, n_layers, ff_dim, max_len, dropout, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, n_heads, ff_dim, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("positions", torch.arange(max_len).unsqueeze(0), persistent=False)

    def forward(self, ids):
        # ids: (B, L). Padding mask marks positions to ignore in attention.
        B, L = ids.shape
        pad_mask = ids.eq(self.pad_id)
        x = self.token_emb(ids) + self.pos_emb(self.positions[:, :L])
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, pad_mask)
        x = self.norm(x)
        return x[:, 0]          # the [CLS] position is index 0 -> (B, D)


class MCQAnswerSelector(nn.Module):
    """Wraps the shared encoder with a linear scorer for four-option MCQ.

    Each of the four (question, option) sequences is encoded independently by the
    same weights; the scorer maps each pooled vector to a scalar, and the four
    scalars form the logits of a softmax over options.
    """

    def __init__(self, vocab_size, pad_id, dim=128, n_heads=4, n_layers=2,
                 ff_dim=256, max_len=96, dropout=0.1, n_options=4):
        super().__init__()
        self.n_options = n_options
        self.encoder = TransformerEncoder(vocab_size, dim, n_heads, n_layers,
                                          ff_dim, max_len, dropout, pad_id)
        self.scorer = nn.Linear(dim, 1)

    def forward(self, ids):
        # ids: (B, n_options, L) -> encode all option-sequences jointly.
        B, O, L = ids.shape
        pooled = self.encoder(ids.view(B * O, L))       # (B*O, D)
        logits = self.scorer(pooled).view(B, O)         # (B, O)
        return logits

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
