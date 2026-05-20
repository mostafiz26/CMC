"""
training.py
-----------
Loss functions and training utilities for CMC's two-phase training.

Phase-1 (Decoder Alignment)
    AE loss: decoder reconstructs the full context from CMEs.
    AR loss: decoder predicts the second half of the context given CMEs
             and the first half.
    L_P1 = L_AE + L_AR

Phase-2 (Answer-Targeted Distillation)
    L_CE   : cross-entropy on gold answer tokens.
    L_KL   : KL divergence between student and Educator LLM logit distributions.
    L_CL   : InfoNCE contrastive loss pulling mean-pooled CMEs toward the
             Educator's answer-span hidden state.
    L_P2 = λ_CE · L_CE + λ_KL · L_KL + λ_CL · L_CL
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .compression import build_memory_vectors
from .config import CMCConfig
from .utils import gold_char_window, build_local_window_tokens


# ── Decoder input packing ─────────────────────────────────────────────────────

def pack_decoder_inputs(
    texts: List[str],
    mem_prefixes: List[torch.Tensor],
    dec_tok,
    dec_model,
    cfg: CMCConfig,
    memory_bridge,
):
    """
    Build decoder input tensors by prepending CME memory prefix to token embeddings.

    Returns
    -------
    inputs_embeds : (B, M_b + T, H_d)
    attn_mask     : (B, M_b + T)
    position_ids  : (B, M_b + T)
    labels        : (B, M_b + T)  — -100 for memory positions
    mb_list       : list of int, memory lengths per example
    raw_ids       : (B, T) token IDs (without memory prefix)
    """
    base_dev = dec_model.get_input_embeddings().weight.device
    enc = dec_tok(
        texts,
        padding=True,
        truncation=True,
        max_length=cfg.dec_max_length,
        return_tensors="pt",
    )
    ids = enc["input_ids"]
    mask = enc["attention_mask"]

    with torch.no_grad():
        tok_emb = dec_model.get_input_embeddings()(ids.to(base_dev))
        tok_emb = torch.nan_to_num(tok_emb, nan=0.0, posinf=1e3, neginf=-1e3).clamp_(-6.0, 6.0)

    B, T, Hd = tok_emb.shape
    mb_list = [mp.size(0) for mp in mem_prefixes]
    max_mb = max(mb_list) if mb_list else 0

    if max_mb > 0:
        mem_pad = torch.zeros((B, max_mb, Hd), device=base_dev, dtype=tok_emb.dtype)
        for b in range(B):
            m = mb_list[b]
            if m > 0:
                mvec = mem_prefixes[b].to(base_dev, dtype=tok_emb.dtype)
                mvec = torch.nan_to_num(mvec, nan=0.0, posinf=1e3, neginf=-1e3).clamp_(-6.0, 6.0)
                norms = mvec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                nu = memory_bridge.scale.abs() * memory_bridge.target_norm
                mvec = mvec * torch.minimum(torch.ones_like(norms), (2.0 * nu) / norms)
                mem_pad[b, :m] = mvec

        inputs_embeds = torch.cat([mem_pad, tok_emb], dim=1)
        attn_mask = torch.cat(
            [torch.ones((B, max_mb), dtype=mask.dtype), mask], dim=1
        ).to(base_dev)
        position_ids = torch.arange(inputs_embeds.size(1), device=base_dev).unsqueeze(0).expand(B, -1)
        labels = torch.cat(
            [torch.full((B, max_mb), -100, dtype=ids.dtype), ids], dim=1
        ).to(base_dev)
    else:
        inputs_embeds = tok_emb
        attn_mask = mask.to(base_dev)
        position_ids = torch.arange(T, device=base_dev).unsqueeze(0).expand(B, -1)
        labels = ids.to(base_dev)

    return inputs_embeds, attn_mask, position_ids, labels, mb_list, ids.to(base_dev)


# ── Safe decoder forward (with numerical fallback) ───────────────────────────

def safe_decoder_forward(inputs_embeds, attention_mask, position_ids, labels, dec_model, max_retries: int = 3):
    for attempt in range(max_retries):
        out = dec_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=False,
            output_hidden_states=True,
        )
        if torch.isfinite(out.loss) and torch.isfinite(out.logits).all():
            return out
        # Scale down memory prefix on instability
        mem_len = (labels[:, : inputs_embeds.size(1)] == -100).all(dim=0).sum().item()
        if attempt < max_retries - 1 and mem_len > 0:
            shrink = 0.5 ** (attempt + 1)
            mem = inputs_embeds[:, :mem_len] * shrink
            inputs_embeds = torch.cat([mem, inputs_embeds[:, mem_len:]], dim=1)
        else:
            break
    return out


# ── Phase-1: AE + AR losses ──────────────────────────────────────────────────

def phase1_ae_ar_losses(
    contexts: List[str],
    comp_tok,
    comp_model,
    memory_bridge,
    dec_tok,
    dec_model,
    cfg: CMCConfig,
    device: torch.device,
    decoder_hidden_dim: int,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute AE and AR reconstruction losses.

    L_AE: decoder reconstructs the full context from CMEs.
    L_AR: decoder predicts the second half given CMEs + first half.
    """
    mem_prefixes = [
        build_memory_vectors(c, comp_tok, comp_model, memory_bridge, cfg, device, decoder_hidden_dim, training)
        for c in contexts
    ]

    # AE loss
    emb, mask, pos, labels, _, _ = pack_decoder_inputs(
        contexts, mem_prefixes, dec_tok, dec_model, cfg, memory_bridge
    )
    out_ae = safe_decoder_forward(emb, mask, pos, labels, dec_model)
    loss_ae = out_ae.loss if torch.isfinite(out_ae.loss) else torch.tensor(float("nan"), device=emb.device)

    # AR loss — predict second half given CMEs + first half
    enc_ctx = dec_tok(
        contexts, padding=True, truncation=True,
        max_length=cfg.dec_max_length, return_tensors="pt"
    )
    ids_ctx = enc_ctx["input_ids"]
    split = max(2, ids_ctx.size(1) // 2)
    prefix_texts = [
        dec_tok.decode(ids_ctx[b, :split], skip_special_tokens=True)
        for b in range(ids_ctx.size(0))
    ]
    emb2, mask2, pos2, labels2, _, _ = pack_decoder_inputs(
        prefix_texts, mem_prefixes, dec_tok, dec_model, cfg, memory_bridge
    )
    out_ar = safe_decoder_forward(emb2, mask2, pos2, labels2, dec_model)
    loss_ar = out_ar.loss if torch.isfinite(out_ar.loss) else torch.tensor(float("nan"), device=emb2.device)

    return loss_ae, loss_ar


# ── Phase-2: QA Knowledge Distillation ───────────────────────────────────────

def _educator_forward(ctx: str, question: str, gold: str, answer_start: int,
                      dec_tok, dec_model, cfg: CMCConfig):
    """Run Educator LLM on a gold-aligned context window."""
    wctx, _ = gold_char_window(ctx, answer_start, cfg.local_window_chars)
    sys_msg = (
        "You are an extractive QA assistant. Return exactly a substring from the context. "
        "Put only the answer between triple quotes."
    )
    user_msg = f"Context:\n{wctx}\n\nQuestion: {question}\nAnswer (in triple quotes):"
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
    prompt = dec_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    dev = dec_model.get_input_embeddings().weight.device
    enc = dec_tok(prompt, return_tensors="pt", truncation=True, max_length=cfg.dec_max_length).to(dev)

    gold_ids = dec_tok(" " + gold, add_special_tokens=False)["input_ids"]
    L = enc["input_ids"].size(1)
    ans_len = min(len(gold_ids), L)

    labels = torch.full_like(enc["input_ids"], -100)
    if ans_len > 0:
        labels[0, -ans_len:] = torch.tensor(gold_ids[-ans_len:], device=dev)

    with torch.no_grad():
        out = dec_model(**enc, labels=labels, output_hidden_states=True, use_cache=False)
        t_logits = out.logits[:, -ans_len:] if ans_len > 0 else out.logits[:, -1:]
        h = out.hidden_states[-1][:, -ans_len:] if ans_len > 0 else out.hidden_states[-1][:, -1:]
        ans_vec = h.mean(dim=1)  # [1, H_d]

    return t_logits.detach(), ans_vec.detach(), ans_len, wctx


def _pack_student_inputs(
    ctx: str, question: str, answer_start: int, drop_local: bool,
    comp_tok, comp_model, memory_bridge, dec_tok, dec_model, cfg: CMCConfig,
    device: torch.device, decoder_hidden_dim: int,
):
    """Build student (CMC) input: [CME prefix] + [local window] + question."""
    mem_full = build_memory_vectors(
        ctx, comp_tok, comp_model, memory_bridge, cfg, device, decoder_hidden_dim, training=False
    )
    # Tier-1 top-K selection by CME norm
    if mem_full.size(0) > cfg.distant_top_k:
        scores = mem_full.norm(dim=-1)
        topk_idx = torch.topk(scores, k=cfg.distant_top_k, largest=True).indices
        mem_full = mem_full[topk_idx]
    mb = mem_full.size(0)

    local_ctx = (
        ""
        if drop_local
        else build_local_window_tokens(ctx, answer_start, cfg.local_window_tokens, dec_tok)
    )

    sys_msg = (
        "You are an extractive QA assistant. Return exactly a short substring from the given text. "
        "Put only the answer between triple quotes."
    )
    user_msg = f"Text:\n{local_ctx}\n\nQuestion: {question}\nAnswer (in triple quotes):"
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
    prompt = dec_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    dev = dec_model.get_input_embeddings().weight.device
    enc_s = dec_tok(prompt, return_tensors="pt", truncation=True, max_length=cfg.dec_max_length)
    ids_s = enc_s["input_ids"].to(dev)
    tok_emb = dec_model.get_input_embeddings()(ids_s)  # [1, T, H_d]

    if mb > 0:
        mvec = mem_full.to(dev, dtype=tok_emb.dtype)
        mvec = torch.nan_to_num(mvec, nan=0.0, posinf=1e3, neginf=-1e3).clamp_(-6.0, 6.0)
        norms = mvec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        nu = memory_bridge.scale.abs() * memory_bridge.target_norm
        mvec = mvec * torch.minimum(torch.ones_like(norms), (2.0 * nu) / norms)
        inputs_embeds = torch.cat([mvec.unsqueeze(0), tok_emb], dim=1)
        attn_mask = torch.ones((1, mb + tok_emb.size(1)), dtype=torch.long, device=dev)
        position_ids = torch.arange(mb + tok_emb.size(1), device=dev).unsqueeze(0)
    else:
        inputs_embeds = tok_emb
        attn_mask = torch.ones((1, tok_emb.size(1)), dtype=torch.long, device=dev)
        position_ids = torch.arange(tok_emb.size(1), device=dev).unsqueeze(0)

    return inputs_embeds, attn_mask, position_ids, mb, local_ctx


def _contrastive_loss(mem_summary: torch.Tensor, ans_vecs: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE: pull each CME mean-pool toward its own Educator answer vector."""
    mem_n = F.normalize(mem_summary, dim=-1)
    ans_n = F.normalize(ans_vecs, dim=-1)
    logits = mem_n @ ans_n.t() / temperature
    targets = torch.arange(mem_n.size(0), device=mem_n.device)
    return F.cross_entropy(logits, targets)


def phase2_kd_loss(
    batch_items: List[Dict],
    comp_tok,
    comp_model,
    memory_bridge,
    dec_tok,
    dec_model,
    cfg: CMCConfig,
    device: torch.device,
    decoder_hidden_dim: int,
) -> Optional[torch.Tensor]:
    """
    Compute the Phase-2 knowledge-distillation loss over a mini-batch.

    Returns None if all examples in the batch are invalid (empty answers, etc.).
    """
    dev = dec_model.get_input_embeddings().weight.device
    ce_losses, kl_losses, mem_summaries, ans_summaries = [], [], [], []

    for ex in batch_items:
        ctx, q, ans, a0 = ex["context"], ex["question"], ex["answer"], ex["answer_start"]
        if not ans.strip():
            continue

        t_logits, t_ans_vec, ans_len, _ = _educator_forward(ctx, q, ans, a0, dec_tok, dec_model, cfg)
        if ans_len == 0:
            continue

        drop_local = torch.rand(()).item() < cfg.drop_local_prob
        stu_emb, stu_mask, stu_pos, mb, local_ctx = _pack_student_inputs(
            ctx, q, a0, drop_local,
            comp_tok, comp_model, memory_bridge, dec_tok, dec_model, cfg, device, decoder_hidden_dim
        )

        gold_ids = dec_tok(" " + ans, add_special_tokens=False)["input_ids"][-ans_len:]
        labels_s = torch.full((1, stu_emb.size(1)), -100, dtype=torch.long, device=dev)
        labels_s[0, -ans_len:] = torch.tensor(gold_ids, device=dev)

        out_s = dec_model(
            inputs_embeds=stu_emb,
            attention_mask=stu_mask,
            position_ids=stu_pos,
            labels=labels_s,
            output_hidden_states=True,
            use_cache=False,
        )
        if not torch.isfinite(out_s.loss):
            continue

        ce_losses.append(out_s.loss)

        # KL divergence
        kl = F.kl_div(
            F.log_softmax(out_s.logits[:, -ans_len:].float(), dim=-1),
            F.softmax(t_logits.float(), dim=-1),
            reduction="batchmean",
        )
        kl_losses.append(kl)

        mem_summary = (
            stu_emb[0, :mb].mean(dim=0, keepdim=True).float()
            if mb > 0
            else torch.zeros((1, stu_emb.size(-1)), device=dev)
        )
        mem_summaries.append(mem_summary)
        ans_summaries.append(t_ans_vec.float().to(dev))

    if not ce_losses:
        return None

    loss = torch.stack(ce_losses).mean()
    if kl_losses:
        loss = loss + cfg.kd_weight_kl * torch.stack(kl_losses).mean()
    if mem_summaries:
        ms = torch.cat(mem_summaries, dim=0)
        av = torch.cat(ans_summaries, dim=0)
        loss = loss + cfg.kd_weight_ctr * _contrastive_loss(
            ms, av, temperature=cfg.contrastive_temperature
        )

    return loss * cfg.kd_weight_ce
