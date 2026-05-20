"""
utils.py
--------
Shared utility functions:
  - Checkpoint save / load
  - Per-layer timing hooks for the DecoderLLM
  - Text windowing helpers used by both training and evaluation
  - Data collection from HuggingFace datasets
"""

from __future__ import annotations
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import torch


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    save_dir: str,
    comp_model,
    comp_tok,
    memory_bridge,
    cfg,
) -> None:
    """
    Persist the trainable components (ContextEncoder + MemoryBridge) and
    the minimal runtime config needed for inference.

    Directory layout
    ----------------
    save_dir/
        compressor/          # ContextEncoder weights + tokenizer
        memory_bridge.pt     # MemoryBridge state_dict
        cmc_config.json      # Inference-time hyperparameters
    """
    os.makedirs(save_dir, exist_ok=True)

    # ContextEncoder
    comp_model.save_pretrained(os.path.join(save_dir, "compressor"))
    comp_tok.save_pretrained(os.path.join(save_dir, "compressor"))

    # MemoryBridge
    torch.save(memory_bridge.state_dict(), os.path.join(save_dir, "memory_bridge.pt"))

    # Inference config (subset of CMCConfig)
    cfg_to_save = {
        "window_k":           cfg.window_k,
        "tau":                cfg.tau,
        "keep_min_every":     cfg.keep_min_every,
        "chunk_tokens":       cfg.chunk_tokens,
        "comp_rate":          cfg.comp_rate,
        "mem_token_cap":      cfg.mem_token_cap,
        "mem_min_tokens":     cfg.mem_min_tokens,
        "comp_max_length":    cfg.comp_max_length,
        "dec_max_length":     cfg.dec_max_length,
        "local_window":       cfg.local_window,
        "distant_top_k":      cfg.distant_top_k,
        "local_window_tokens":cfg.local_window_tokens,
        "mem_b":              "<MEM>",
        "mem_e":              "</MEM>",
        "decoder_model":      cfg.decoder_model,
        "llama_4bit":         cfg.llama_4bit,
        "llama_dtype":        cfg.llama_dtype,
    }
    with open(os.path.join(save_dir, "cmc_config.json"), "w") as f:
        json.dump(cfg_to_save, f, indent=2)

    print(f"[CMC] Checkpoint saved to: {save_dir}")


def load_checkpoint(save_dir: str) -> Dict:
    """
    Load inference config from a saved checkpoint directory.

    Returns
    -------
    dict with the saved hyperparameter values.
    """
    config_path = os.path.join(save_dir, "cmc_config.json")
    with open(config_path) as f:
        return json.load(f)


# ── Per-layer timing hooks for the DecoderLLM ────────────────────────────────

# Shared mutable dict populated by hooks
layer_times: Dict[int, Dict] = {}


def make_timing_hooks(dec_model) -> List:
    """
    Attach forward pre/post hooks to every DecoderLLM transformer block.
    Elapsed time per layer is written into `layer_times`.

    Returns
    -------
    List of hook handles (pass to remove_timing_hooks when done).
    """
    handles = []
    for layer_idx, block in enumerate(dec_model.model.layers):

        def pre_hook(idx):
            def _pre(_module, _inputs):
                layer_times[idx] = {"start": time.time(), "elapsed": 0.0}
            return _pre

        def post_hook(idx):
            def _post(_module, _inputs, _output):
                layer_times[idx]["elapsed"] = time.time() - layer_times[idx]["start"]
            return _post

        handles.append(block.register_forward_pre_hook(pre_hook(layer_idx)))
        handles.append(block.register_forward_hook(post_hook(layer_idx)))

    return handles


def remove_timing_hooks(handles: List) -> None:
    for h in handles:
        h.remove()


# ── Text windowing helpers ────────────────────────────────────────────────────

def gold_char_window(
    ctx: str, answer_start: int, window_chars: int
) -> Tuple[str, int]:
    """
    Extract a character window of width `window_chars` centred around
    `answer_start`.  Returns (window_text, window_start_char_offset).
    """
    L = len(ctx)
    if L <= window_chars:
        return ctx, 0
    start = max(0, min(L - window_chars, answer_start - window_chars // 2))
    return ctx[start : start + window_chars], start


def build_local_window_tokens(
    ctx: str, answer_start: int, max_tok: int, dec_tok
) -> str:
    """
    Build the Tier-2 local context window for the student (CMC).
    Extracts a 512-character window around the answer and then caps at
    `max_tok` decoder tokens.
    """
    loc_ctx, _ = gold_char_window(ctx, answer_start, window_chars=512)
    tok_enc = dec_tok(
        loc_ctx,
        add_special_tokens=True,
        truncation=True,
        max_length=max_tok,
        return_tensors="pt",
    )
    return dec_tok.decode(tok_enc["input_ids"][0].tolist(), skip_special_tokens=True)


# ── Data collection ───────────────────────────────────────────────────────────

def collect_examples(split, n: int, shuffle: bool = True) -> List[Dict]:
    """
    Collect at most `n` examples from a HuggingFace dataset split.
    Each example is a dict with keys: context, question, answer, answer_start.

    Parameters
    ----------
    split : HuggingFace dataset split object
    n : int
        Maximum number of examples to return.
    shuffle : bool
        If True, shuffle before truncating to n.
    """
    import random

    items = []
    for ex in split:
        if len(ex["answers"]["text"]) == 0:
            continue
        items.append(
            dict(
                context=ex["context"],
                question=ex["question"],
                answer=ex["answers"]["text"][0],
                answer_start=ex["answers"]["answer_start"][0],
            )
        )
    if shuffle:
        random.shuffle(items)
    return items[:n]
