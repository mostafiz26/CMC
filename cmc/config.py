"""
Configuration dataclass for the CMC framework.
All hyperparameters are centralised here so that experiments can be
reproduced by pointing to a single config file or overriding fields
on the command line via train.py.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class CMCConfig:
    # ── Reproducibility ─────────────────────────────────────────────────────
    seed: int = 42

    # ── Graph-based context denoising ───────────────────────────────────────
    # Neighbourhood window for cosine-similarity graph edges
    window_k: int = 3
    # Similarity threshold τ: tokens with sim(e_i, e_{i+1}) < τ are retained
    tau: float = 0.05
    # Forced keep-every interval (floor constraint)
    keep_min_every: int = 16
    # Warm-up steps before denoising is activated
    ig_warmup_steps: int = 200

    # ── Memory / compression ────────────────────────────────────────────────
    chunk_tokens: int = 256          # segment length t
    comp_rate: int = 4               # compression rate r = t / m(c)
    mem_token_cap: int = 64          # maximum memory tokens per chunk
    mem_min_tokens: int = 4          # minimum memory tokens per chunk
    warmup_fixed_m: int = 8          # fixed m used during IG warm-up

    # ── Models ──────────────────────────────────────────────────────────────
    # ContextEncoder choices: gpt2-large | facebook/opt-1.3b | facebook/opt-2.7b
    compressor_model: str = "gpt2-large"
    # DecoderLLM choices: meta-llama/Meta-Llama-3-8B-Instruct |
    #                     mistralai/Mistral-7B-Instruct-v0.3   |
    #                     google/gemma-2-9b-it
    decoder_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    # ── Decoder quantisation ─────────────────────────────────────────────────
    llama_4bit: bool = True
    llama_dtype: str = "bfloat16"    # "bfloat16" or "float16"
    llama_device_map: str = "auto"
    freeze_decoder: bool = True

    # ── Sequence length caps ─────────────────────────────────────────────────
    comp_max_length: int = 3072
    dec_max_length: int = 1536
    max_new_tokens: int = 32

    # ── Phase-1 training (AE / AR) ───────────────────────────────────────────
    epochs: int = 1
    batch_size: int = 8
    lr: float = 1e-5
    wd: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.98)
    max_grad_norm: float = 0.6

    # ── Phase-2 QA knowledge distillation ───────────────────────────────────
    qa_kd_steps_per_epoch: int = 800
    qa_batch_size: int = 2
    kd_weight_ce: float = 1.0        # CE loss weight  λ_CE
    kd_weight_kl: float = 0.5        # KL loss weight  λ_KL
    kd_weight_ctr: float = 0.1       # Contrastive loss weight λ_CL
    contrastive_temperature: float = 0.07
    drop_local_prob: float = 0.5     # probability of dropping Tier-2 window during KD
    local_window_chars: int = 1200   # Educator LLM gold-aligned character window
    local_window_tokens: int = 130   # Student Tier-2 window in decoder tokens

    # ── Two-tier KV cache inference policy ──────────────────────────────────
    local_window: int = 130          # Tier-2 window width W (tokens)
    distant_top_k: int = 30          # Tier-1 top-K CMEs

    # ── Dataset sizes ────────────────────────────────────────────────────────
    train_samples: int = 86_000
    val_samples: int = 6_000
    shuffle: bool = True

    # ── Output ───────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints/cmc"
