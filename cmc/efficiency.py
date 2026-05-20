"""
efficiency.py
-------------
Analytical estimates of attention MACs and KV-cache memory for CMC
versus the uncompressed-context baseline.

These functions reproduce the theoretical efficiency analysis reported
in the paper (Appendix A.5).  All estimates are hardware-independent —
they depend only on model architecture parameters and the KV cache
policy, not on measured wall-clock time.
"""

from __future__ import annotations
from typing import Dict

from .config import CMCConfig


# ── Attention MACs ────────────────────────────────────────────────────────────

def _attn_macs_per_step(n_heads: int, head_dim: int, past_len: int) -> int:
    """
    Dot-product attention MACs for a single generation step with KV-cache
    length past_len.  Each head computes n_heads × head_dim × past_len
    multiply-accumulate operations for the score step (×2 for the
    value-aggregation step, giving 2 total).
    """
    return 2 * n_heads * head_dim * past_len


def _kv_len_baseline(step: int, prompt_len: int) -> int:
    return prompt_len + step


def _kv_len_cmc(step: int, prompt_len: int, mem_tokens: int,
                local_window: int, distant_top_k: int) -> int:
    """
    Effective KV-cache length under the two-tier policy.
    Never exceeds min(local_window, prompt_len + step) + min(distant_top_k, mem_tokens).
    """
    tier2 = min(local_window, prompt_len + step)
    tier1 = min(distant_top_k, mem_tokens)
    return tier2 + tier1


def estimate_total_attn_macs(
    n_layers: int,
    n_heads: int,
    head_dim: int,
    prompt_len: int,
    gen_len: int,
    mem_tokens: int,
    cfg: CMCConfig,
) -> Dict[str, int]:
    """
    Estimate total attention MACs for both the baseline and CMC over a
    full autoregressive generation of `gen_len` tokens.

    Parameters
    ----------
    n_layers, n_heads, head_dim : decoder architecture parameters
    prompt_len : number of prompt tokens fed at prefill
    gen_len    : number of tokens to generate
    mem_tokens : Mb — total CMEs produced for this context
    cfg        : CMCConfig (provides local_window and distant_top_k)

    Returns
    -------
    dict with keys: baseline_macs, cmc_macs, reduction_pct
    """
    baseline_total = 0
    cmc_total = 0

    for t in range(gen_len):
        kv_base = _kv_len_baseline(t, prompt_len)
        kv_cmc = _kv_len_cmc(
            t, prompt_len, mem_tokens, cfg.local_window, cfg.distant_top_k
        )
        per_layer_base = _attn_macs_per_step(n_heads, head_dim, kv_base)
        per_layer_cmc  = _attn_macs_per_step(n_heads, head_dim, kv_cmc)
        baseline_total += per_layer_base * n_layers
        cmc_total += per_layer_cmc * n_layers

    reduction = 100.0 * (1.0 - cmc_total / max(1, baseline_total))
    return {
        "baseline_macs": baseline_total,
        "cmc_macs":      cmc_total,
        "reduction_pct": round(reduction, 2),
    }


# ── KV-cache memory ───────────────────────────────────────────────────────────

def kv_cache_bytes_per_token(d_model: int, dtype_bytes: int = 2) -> int:
    """Bytes per token per layer for a single K or V tensor (combined K+V = ×2)."""
    return 2 * d_model * dtype_bytes


def estimate_kv_cache_memory(
    n_layers: int,
    tokens_cached: int,
    d_model: int,
    dtype_bytes: int = 2,
) -> int:
    """Total KV-cache bytes across all layers."""
    return n_layers * tokens_cached * kv_cache_bytes_per_token(d_model, dtype_bytes)


def kv_cache_report(
    n_layers: int,
    n_heads: int,
    head_dim: int,
    d_model: int,
    prompt_len_baseline: int,
    prompt_len_cmc: int,
    mem_tokens: int,
    cfg: CMCConfig,
) -> Dict:
    """
    Full KV-cache and MACs report comparable to Table 16 in the paper.

    Parameters
    ----------
    prompt_len_baseline : baseline prefill token count (context + question)
    prompt_len_cmc      : CMC prefill token count (local_window + question)
    mem_tokens          : Mb for this example
    """
    T = cfg.max_new_tokens
    dtype_bytes = 2  # bfloat16 / float16

    macs = estimate_total_attn_macs(
        n_layers, n_heads, head_dim,
        prompt_len_cmc, T, mem_tokens, cfg,
    )
    # For baseline MACs, reuse prompt_len_baseline with no KV cap
    baseline_macs_detail = sum(
        _attn_macs_per_step(n_heads, head_dim, _kv_len_baseline(t, prompt_len_baseline)) * n_layers
        for t in range(T)
    )

    kv_base = estimate_kv_cache_memory(n_layers, prompt_len_baseline + T, d_model, dtype_bytes)
    kv_cmc  = estimate_kv_cache_memory(n_layers, prompt_len_cmc + T, d_model, dtype_bytes)

    return {
        "prompt_lengths": {
            "baseline_tokens": prompt_len_baseline,
            "cmc_prompt_tokens": prompt_len_cmc,
            "memory_vectors_Mb": mem_tokens,
            "gen_tokens_T": T,
        },
        "attn_macs": {
            "baseline": baseline_macs_detail,
            "cmc":      macs["cmc_macs"],
            "reduction_pct": round(100.0 * (1 - macs["cmc_macs"] / max(1, baseline_macs_detail)), 2),
        },
        "kv_cache_bytes": {
            "per_token_per_layer": kv_cache_bytes_per_token(d_model, dtype_bytes),
            "baseline_total": kv_base,
            "cmc_total":      kv_cmc,
            "baseline_MB": round(kv_base / 1024 / 1024, 2),
            "cmc_MB":      round(kv_cmc  / 1024 / 1024, 2),
            "reduction_pct": round(100.0 * (1 - kv_cmc / max(1, kv_base)), 2),
        },
        "policy": {
            "local_window_W":  cfg.local_window,
            "top_k_K":         cfg.distant_top_k,
        },
    }
