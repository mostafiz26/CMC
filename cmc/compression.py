"""
compression.py
--------------
Context denoising, chunking, and CME (Context Memory Embedding) extraction.

Graph-based context denoising
    Tokens whose cosine similarity to the next adjacent token falls below
    threshold τ are retained; highly redundant tokens are dropped.  A
    keep_min_every floor prevents over-pruning on very smooth passages.

CME extraction
    Each denoised context is split into fixed-length segments of t tokens.
    Special <MEM> placeholder tokens are appended to each segment and
    processed by the ContextEncoder (a causal LM).  The hidden states of
    the placeholder positions in the final transformer layer form the
    segment-level CMEs.  These are then projected into the decoder
    embedding space via the MemoryBridge.
"""

from __future__ import annotations
from typing import List

import torch
import torch.nn.functional as F

from .config import CMCConfig


# ── Global train-step counter (incremented by the training loop) ─────────────
_global_step: int = 0


def set_global_step(step: int) -> None:
    global _global_step
    _global_step = step


def get_global_step() -> int:
    return _global_step


# ── Graph-based token denoising ───────────────────────────────────────────────

@torch.no_grad()
def ig_prune_token_ids(
    token_ids: List[int],
    emb_table: torch.Tensor,
    k: int,
    tau: float,
    keep_every: int,
) -> List[int]:
    """
    Graph-based context denoising.

    Retains tokens whose embedding cosine-similarity to a neighbour
    within window k exceeds threshold τ.  A forced keep_every floor
    ensures a minimum token density.

    Parameters
    ----------
    token_ids : list of int
        Token IDs of the full context (after tokenisation).
    emb_table : Tensor of shape (vocab_size, H_c)
        Embedding table of the ContextEncoder.
    k : int
        Neighbourhood window size.
    tau : float
        Cosine-similarity threshold.
    keep_every : int
        Forced keep-every interval (floor constraint).

    Returns
    -------
    list of int
        Pruned token ID list.
    """
    if len(token_ids) <= 2:
        return token_ids

    dev = emb_table.device
    idx = torch.tensor(token_ids, device=dev, dtype=torch.long)
    vecs = F.normalize(emb_table[idx].float(), dim=-1)
    N = vecs.size(0)

    keep = torch.zeros(N, dtype=torch.bool, device=dev)
    for i in range(N - 1):
        end = min(N, i + k + 1)
        nbrs = vecs[i + 1 : end]
        if nbrs.size(0) == 0:
            break
        sims = torch.mv(nbrs, vecs[i])
        if (sims >= tau).any():
            keep[i] = True
            keep[i + 1 : end][sims >= tau] = True

    # Floor constraint
    keep[:: max(1, keep_every)] = True
    keep[0] = True
    keep[-1] = True

    kept = [token_ids[i] for i in range(N) if bool(keep[i].item())]

    # Safety fallback: ensure minimum token count
    if len(kept) < max(16, N // (2 * keep_every)):
        step = max(1, N // max(16, N // keep_every))
        kept = token_ids[::step]
        if kept[-1] != token_ids[-1]:
            kept.append(token_ids[-1])

    return kept


# ── Chunking helpers ──────────────────────────────────────────────────────────

def chunk_list(xs: List[int], L: int) -> List[List[int]]:
    return [xs[i : i + L] for i in range(0, len(xs), L)]


def m_for_chunk(n: int, r: int, cap: int, mmin: int) -> int:
    """Number of placeholder tokens for a chunk of length n."""
    return min(max(mmin, n // r), cap)


def append_mem_placeholders(
    ids: List[int], m: int, mem_b: int, mem_e: int, placeholder: int
) -> List[int]:
    return ids + [mem_b] + [placeholder] * m + [mem_e]


# ── CME extraction ────────────────────────────────────────────────────────────

def build_memory_vectors(
    context: str,
    comp_tok,
    comp_model,
    memory_bridge,
    cfg: CMCConfig,
    device: torch.device,
    decoder_hidden_dim: int,
    training: bool = False,
) -> torch.Tensor:
    """
    Compress a context string into a matrix of decoder-space CME vectors.

    Steps
    -----
    1. Tokenise the context.
    2. Apply graph-based denoising (after warm-up steps).
    3. Split into fixed-length segments of cfg.chunk_tokens tokens.
    4. Append <MEM>…</MEM> placeholders to each segment.
    5. Run the ContextEncoder; extract placeholder hidden states.
    6. Project each CME via the MemoryBridge into decoder space.
    7. Concatenate all segment CMEs into a single matrix M ∈ ℝ^{M_b × H_d}.

    Parameters
    ----------
    context : str
    comp_tok : HuggingFace tokenizer for the ContextEncoder.
    comp_model : ContextEncoder (trainable causal LM).
    memory_bridge : MemoryBridge module.
    cfg : CMCConfig
    device : torch.device (CUDA / CPU)
    decoder_hidden_dim : int  (H_d)
    training : bool
        If True, comp_model is in train mode and gradients flow through it.

    Returns
    -------
    Tensor of shape (M_b, H_d) on `device`.
    """
    MEM_B = "<MEM>"
    MEM_E = "</MEM>"

    enc = comp_tok(
        context,
        add_special_tokens=True,
        truncation=True,
        max_length=cfg.comp_max_length,
        return_tensors="pt",
    )
    ids: List[int] = enc["input_ids"][0].tolist()

    # Apply denoising only after warm-up
    use_ig = (not training) or (_global_step >= cfg.ig_warmup_steps)
    if use_ig:
        with torch.no_grad():
            emb_table = comp_model.get_input_embeddings().weight.data
        ids = ig_prune_token_ids(
            ids, emb_table, cfg.window_k, cfg.tau, cfg.keep_min_every
        )

    chunks = chunk_list(ids, cfg.chunk_tokens)
    if not chunks:
        return torch.zeros((0, decoder_hidden_dim), device=device, dtype=torch.float32)

    mem_b_id = comp_tok.convert_tokens_to_ids(MEM_B)
    mem_e_id = comp_tok.convert_tokens_to_ids(MEM_E)  # noqa: F841 (reserved for future use)
    unk_id = comp_tok.unk_token_id

    mem_vecs = []
    comp_model.train(training)

    for ch in chunks:
        m = (
            m_for_chunk(len(ch), cfg.comp_rate, cfg.mem_token_cap, cfg.mem_min_tokens)
            if use_ig
            else min(max(cfg.mem_min_tokens, cfg.warmup_fixed_m), cfg.mem_token_cap)
        )
        ch_mem = append_mem_placeholders(ch, m, mem_b_id, mem_e_id, unk_id)
        t_ids = torch.tensor([ch_mem], dtype=torch.long, device=device)
        t_att = torch.ones_like(t_ids)

        out = comp_model(
            input_ids=t_ids,
            attention_mask=t_att,
            output_hidden_states=True,
            use_cache=False,
        )
        last = out.hidden_states[-1]           # [1, T, H_c]
        mem_start = len(ch) + 1
        mem_end = mem_start + m
        mem_states = last[0, mem_start:mem_end, :].float()
        mem_states = torch.nan_to_num(
            mem_states, nan=0.0, posinf=1e3, neginf=-1e3
        ).clamp_(-6.0, 6.0)

        mem_dec = memory_bridge(mem_states)    # [m, H_d]
        mem_vecs.append(mem_dec)

    if mem_vecs:
        return torch.cat(mem_vecs, dim=0)
    return torch.zeros((0, decoder_hidden_dim), device=device, dtype=torch.float32)


# ── KV cache sliding window ───────────────────────────────────────────────────

def crop_past_with_memory(past_key_values, mem_len: int, local_window: int):
    """
    Sliding KV-cache crop for CMC inference.

    Retains:
      - First `mem_len` positions  → Tier-1 CME prefix (never evicted).
      - Last `local_window` non-memory positions → Tier-2 sliding window.

    Parameters
    ----------
    past_key_values : tuple of (k, v) per layer
        k, v : Tensor of shape (B, n_heads, T, head_dim)
    mem_len : int
    local_window : int

    Returns
    -------
    Cropped past_key_values tuple.
    """
    if past_key_values is None or mem_len <= 0 or local_window <= 0:
        return past_key_values

    new_past = []
    for k, v in past_key_values:
        T = k.size(2)
        max_len = mem_len + local_window
        if T <= max_len:
            new_past.append((k, v))
            continue

        tail_len = min(local_window, T - mem_len)
        if tail_len <= 0:
            keep_idx = torch.arange(0, mem_len, device=k.device)
        else:
            keep_mem = torch.arange(0, mem_len, device=k.device)
            keep_tail = torch.arange(T - tail_len, T, device=k.device)
            keep_idx = torch.cat([keep_mem, keep_tail], dim=0)

        new_past.append((k.index_select(2, keep_idx), v.index_select(2, keep_idx)))

    return tuple(new_past)
