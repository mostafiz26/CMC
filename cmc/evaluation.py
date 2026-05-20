"""
evaluation.py
-------------
Evaluation utilities for CMC: EM / F1 metrics and answer generation.

Two evaluation paths
--------------------
Baseline
    The frozen DecoderLLM receives the Tier-2 gold-aligned context window
    directly (no compression) and generates an answer autoregressively.

CMC
    The frozen DecoderLLM receives [CME prefix | Tier-2 window | question]
    as inputs_embeds and generates autoregressively with a sliding KV cache
    that never grows beyond K + W tokens.
"""

from __future__ import annotations
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch

from .compression import build_memory_vectors, crop_past_with_memory
from .config import CMCConfig
from .utils import gold_char_window, build_local_window_tokens
from .training import _pack_student_inputs


# ── String normalisation and EM / F1 ─────────────────────────────────────────

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def em_f1(gold: str, pred: str) -> Tuple[float, float]:
    """Return (exact-match, token-F1) for a single prediction."""
    g = normalize_answer(gold)
    p = normalize_answer(pred)
    em = 1.0 if (g == p and g != "") else 0.0
    gt, pt = g.split(), p.split()
    if not gt and not pt:
        return 1.0, 1.0
    common = Counter(gt) & Counter(pt)
    num_same = sum(common.values())
    if num_same == 0:
        return em, 0.0
    prec = num_same / max(1, len(pt))
    rec = num_same / max(1, len(gt))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return em, f1


# ── Answer extraction helpers ────────────────────────────────────────────────

def extract_triple_quotes(text: str) -> str:
    """Extract answer between triple-quote delimiters, or return the full text."""
    m = re.search(r'"""(.*?)"""', text, flags=re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def snap_to_substring(pred: str, window_ctx: str) -> str:
    """Find the longest sub-span of pred that appears verbatim in window_ctx."""
    p = pred.strip().strip('"').strip()
    if p in window_ctx:
        return p
    toks = p.split()
    best = ""
    for i in range(len(toks)):
        for j in range(i + 1, len(toks) + 1):
            cand = " ".join(toks[i:j])
            if cand and cand in window_ctx and len(cand) > len(best):
                best = cand
    return best if best else p


# ── Generation: baseline (no compression) ────────────────────────────────────

@torch.no_grad()
def greedy_generate_ids(
    input_ids: torch.Tensor,
    dec_model,
    dec_tok,
    max_new: int,
) -> str:
    dev = dec_model.get_input_embeddings().weight.device
    out = dec_model.generate(
        input_ids=input_ids.to(dev),
        max_new_tokens=max_new,
        do_sample=False,
        eos_token_id=dec_tok.eos_token_id,
        pad_token_id=dec_tok.pad_token_id,
        repetition_penalty=1.05,
    )
    gen = out[0][input_ids.size(1):]
    return dec_tok.decode(gen, skip_special_tokens=True)


# ── Generation: CMC with sliding KV cache ────────────────────────────────────

@torch.no_grad()
def greedy_generate_from_embeds(
    inputs_embeds: torch.Tensor,
    attn_mask: torch.Tensor,
    pos_ids: torch.Tensor,
    mem_len: int,
    dec_model,
    dec_tok,
    cfg: CMCConfig,
    max_new: Optional[int] = None,
) -> str:
    """
    CMC autoregressive generation with sliding KV-cache policy.

    The KV cache is never larger than mem_len + cfg.local_window tokens at
    any step, regardless of how many tokens have been generated.
    """
    if max_new is None:
        max_new = cfg.max_new_tokens

    dev = dec_model.get_input_embeddings().weight.device
    eos_id = dec_tok.eos_token_id
    B = inputs_embeds.size(0)

    cur_emb = inputs_embeds.to(dev)
    cur_mask = attn_mask.to(dev)
    cur_pos = pos_ids.to(dev)
    past = None

    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    generated_ids: List[List[int]] = [[] for _ in range(B)]

    for _ in range(max_new):
        out = dec_model(
            inputs_embeds=cur_emb,
            attention_mask=cur_mask,
            position_ids=cur_pos,
            use_cache=True,
            past_key_values=past,
        )
        next_ids = out.logits[:, -1].argmax(dim=-1)
        past = out.past_key_values

        if mem_len > 0 and past is not None:
            past = crop_past_with_memory(past, mem_len, cfg.local_window)

        for b in range(B):
            if not finished[b]:
                generated_ids[b].append(int(next_ids[b].item()))
                if next_ids[b].item() == eos_id:
                    finished[b] = True

        if finished.all():
            break

        cur_emb = dec_model.get_input_embeddings()(next_ids.unsqueeze(-1))
        cur_mask = torch.ones((B, 1), dtype=torch.long, device=dev)
        cur_pos = cur_pos[:, -1:] + 1

    ids_0 = generated_ids[0]
    return dec_tok.decode(ids_0, skip_special_tokens=True) if ids_0 else ""


# ── Full QA evaluation ────────────────────────────────────────────────────────

def build_baseline_prompt(window_ctx: str, question: str, dec_tok) -> str:
    sys_msg = "You are a QA system. Read the passage and answer concisely."
    user_msg = f"PASSAGE:\n{window_ctx}\n\nQUESTION:\n{question}\n\nANSWER:"
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
    return dec_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def evaluate_qa(
    examples: List[Dict],
    comp_tok,
    comp_model,
    memory_bridge,
    dec_tok,
    dec_model,
    cfg: CMCConfig,
    device: torch.device,
    decoder_hidden_dim: int,
    limit: Optional[int] = None,
    print_every: int = 50,
) -> Dict:
    """
    Evaluate both the baseline (gold-aligned window, no compression) and
    CMC (CME prefix + Tier-2 window) on EM and F1.

    Returns
    -------
    dict with keys: count, baseline {EM, F1}, cmc {EM, F1}
    """
    n = len(examples) if limit is None else min(limit, len(examples))
    base_em = base_f1 = cmc_em = cmc_f1 = 0.0
    cnt = 0

    for idx in range(n):
        ex = examples[idx]
        ctx, q, ans, a0 = ex["context"], ex["question"], ex["answer"], ex["answer_start"]

        # ── Baseline ─────────────────────────────────────────────────────────
        wctx, _ = gold_char_window(ctx, a0, cfg.local_window_chars)
        prompt_b = build_baseline_prompt(wctx, q, dec_tok)
        enc_b = dec_tok(
            prompt_b, return_tensors="pt", truncation=True, max_length=cfg.dec_max_length
        )
        out_b = greedy_generate_ids(enc_b["input_ids"], dec_model, dec_tok, cfg.max_new_tokens)
        pred_b = snap_to_substring(extract_triple_quotes(out_b), wctx)
        eb, fb = em_f1(ans, pred_b)

        # ── CMC ──────────────────────────────────────────────────────────────
        stu_emb, stu_mask, stu_pos, mb, local_ctx = _pack_student_inputs(
            ctx, q, a0, drop_local=False,
            comp_tok=comp_tok, comp_model=comp_model, memory_bridge=memory_bridge,
            dec_tok=dec_tok, dec_model=dec_model, cfg=cfg,
            device=device, decoder_hidden_dim=decoder_hidden_dim,
        )
        out_c = greedy_generate_from_embeds(
            stu_emb, stu_mask, stu_pos, mb, dec_model, dec_tok, cfg
        )
        pred_c = snap_to_substring(extract_triple_quotes(out_c), local_ctx)
        ec, fc = em_f1(ans, pred_c)

        base_em += eb; base_f1 += fb
        cmc_em += ec; cmc_f1 += fc
        cnt += 1

        if (idx + 1) % print_every == 0:
            print(
                f"[{idx+1}/{n}] baseline EM/F1={base_em/cnt:.3f}/{base_f1/cnt:.3f}"
                f" | cmc EM/F1={cmc_em/cnt:.3f}/{cmc_f1/cnt:.3f}"
            )

    return {
        "count": cnt,
        "baseline": {"EM": round(base_em / cnt, 4), "F1": round(base_f1 / cnt, 4)},
        "cmc":      {"EM": round(cmc_em  / cnt, 4), "F1": round(cmc_f1  / cnt, 4)},
    }
