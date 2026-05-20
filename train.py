"""
train.py
--------
Main entry point for CMC training and evaluation.

Usage
-----
# Train with defaults (SQuAD, GPT2-Large + Llama-3-8B):
    python train.py

# Override hyperparameters:
    python train.py --comp_rate 8 --epochs 2 --checkpoint_dir my_run

# Use a different encoder/decoder pair:
    python train.py \
        --compressor_model facebook/opt-1.3b \
        --decoder_model mistralai/Mistral-7B-Instruct-v0.3
"""

import argparse
import json
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

from cmc import (
    CMCConfig,
    MemoryBridge,
    collect_examples,
    evaluate_qa,
    kv_cache_report,
    phase1_ae_ar_losses,
    phase2_kd_loss,
    save_checkpoint,
)
from cmc.compression import set_global_step, get_global_step
from cmc.utils import gold_char_window, build_local_window_tokens, make_timing_hooks


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> CMCConfig:
    parser = argparse.ArgumentParser(description="Train the CMC framework.")
    # Expose the most commonly varied hyperparameters as CLI flags.
    # All other CMCConfig fields keep their defaults.
    parser.add_argument("--compressor_model", type=str, default=None)
    parser.add_argument("--decoder_model", type=str, default=None)
    parser.add_argument("--comp_rate", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--val_samples", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = CMCConfig()
    for field_name, value in vars(args).items():
        if value is not None and hasattr(cfg, field_name):
            setattr(cfg, field_name, value)
    return cfg


# ── Model loading ─────────────────────────────────────────────────────────────

MEM_B, MEM_E = "<MEM>", "</MEM>"


def load_models(cfg: CMCConfig, device: torch.device):
    """Load ContextEncoder, DecoderLLM, and initialise MemoryBridge."""
    print("Loading tokenizers …")
    comp_tok = AutoTokenizer.from_pretrained(cfg.compressor_model, use_fast=True)
    if comp_tok.pad_token is None:
        comp_tok.pad_token = comp_tok.eos_token
    added = comp_tok.add_special_tokens({"additional_special_tokens": [MEM_B, MEM_E]})
    print(f"  ContextEncoder tokenizer: added {added} special tokens.")

    dec_tok = AutoTokenizer.from_pretrained(cfg.decoder_model, use_fast=True)
    if dec_tok.pad_token is None:
        dec_tok.pad_token = dec_tok.eos_token

    print(f"Loading ContextEncoder ({cfg.compressor_model}) …")
    comp_model = AutoModelForCausalLM.from_pretrained(
        cfg.compressor_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=None,
    )
    if added > 0:
        comp_model.resize_token_embeddings(len(comp_tok))
    comp_model = comp_model.to(device)

    # Infer hidden dimension (GPT-2 uses n_embd; OPT uses hidden_size)
    comp_cfg = comp_model.config
    if hasattr(comp_cfg, "n_embd"):
        Hc = comp_cfg.n_embd
    elif hasattr(comp_cfg, "hidden_size"):
        Hc = comp_cfg.hidden_size
    elif hasattr(comp_cfg, "word_embed_proj_dim"):
        Hc = comp_cfg.word_embed_proj_dim
    else:
        raise ValueError(f"Cannot infer encoder hidden size from config: {comp_cfg}")
    print(f"  ContextEncoder hidden size: {Hc}")

    print(f"Loading DecoderLLM ({cfg.decoder_model}, frozen, 4-bit NF4) …")
    bnb_cfg = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.llama_dtype == "bfloat16" else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        if cfg.llama_4bit
        else None
    )
    dec_model = AutoModelForCausalLM.from_pretrained(
        cfg.decoder_model,
        quantization_config=bnb_cfg,
        torch_dtype=(
            (torch.bfloat16 if cfg.llama_dtype == "bfloat16" else torch.float16)
            if not cfg.llama_4bit
            else None
        ),
        device_map=cfg.llama_device_map,
    )

    Hd = dec_model.config.hidden_size
    n_layers = dec_model.config.num_hidden_layers
    n_heads = dec_model.config.num_attention_heads
    head_dim = Hd // n_heads
    print(
        f"  DecoderLLM: hidden={Hd} | layers={n_layers} | heads={n_heads} | head_dim={head_dim}"
    )

    if cfg.freeze_decoder:
        for p in dec_model.parameters():
            p.requires_grad_(False)
        dec_model.eval()

    # Initialise MemoryBridge with target norm from frozen decoder embeddings
    with torch.no_grad():
        avg_norm = (
            dec_model.get_input_embeddings().weight.detach().norm(dim=-1).mean().item()
        )
    memory_bridge = MemoryBridge(Hc, Hd, target_norm=avg_norm).to(device).float()

    return comp_tok, dec_tok, comp_model, dec_model, memory_bridge, Hd, n_layers, n_heads, head_dim


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: CMCConfig) -> None:
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    comp_tok, dec_tok, comp_model, dec_model, memory_bridge, Hd, n_layers, n_heads, head_dim = load_models(cfg, device)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading dataset …")
    ds = load_dataset("squad")
    train_examples = collect_examples(ds["train"], cfg.train_samples, shuffle=cfg.shuffle)
    val_examples   = collect_examples(ds["validation"], cfg.val_samples, shuffle=False)
    print(f"  Train: {len(train_examples)}  |  Val: {len(val_examples)}")

    # ── Optimiser (ContextEncoder + MemoryBridge only) ────────────────────────
    params = list(comp_model.parameters()) + list(memory_bridge.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.wd)

    losses_log = []
    total_start = time.time()

    for ep in range(cfg.epochs):
        ep_start = time.time()
        if cfg.shuffle:
            random.shuffle(train_examples)

        # ── Phase-1: AE / AR pass ─────────────────────────────────────────────
        for i in range(0, len(train_examples), cfg.batch_size):
            batch = train_examples[i : i + cfg.batch_size]
            contexts = [b["context"] for b in batch]
            opt.zero_grad(set_to_none=True)

            try:
                loss_ae, loss_ar = phase1_ae_ar_losses(
                    contexts, comp_tok, comp_model, memory_bridge,
                    dec_tok, dec_model, cfg, device, Hd, training=True,
                )
                loss_lm = 0.5 * loss_ae + 0.5 * loss_ar
            except RuntimeError as exc:
                torch.cuda.empty_cache()
                print(f"[Phase-1 skip] {str(exc)[:100]}")
                set_global_step(get_global_step() + 1)
                continue

            if not torch.isfinite(loss_lm):
                set_global_step(get_global_step() + 1)
                continue

            loss_lm.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step()
            losses_log.append(float(loss_lm.item()))
            set_global_step(get_global_step() + 1)

            if (i // cfg.batch_size) % 20 == 0:
                avg50 = sum(losses_log[-50:]) / max(1, len(losses_log[-50:]))
                print(
                    f"\r[Phase-1] Epoch {ep+1} | step {i//cfg.batch_size:4d} | loss(avg50) {avg50:.4f}",
                    end="",
                )

        print(f"\n[Phase-1] Epoch {ep+1} complete.")

        # ── Phase-2: QA Knowledge Distillation ───────────────────────────────
        print(f"[Phase-2] Running {cfg.qa_kd_steps_per_epoch} KD steps …")
        for kd_step in range(cfg.qa_kd_steps_per_epoch):
            batch_items = [
                train_examples[(kd_step * cfg.qa_batch_size + j) % len(train_examples)]
                for j in range(cfg.qa_batch_size)
            ]
            opt.zero_grad(set_to_none=True)
            kd_loss = phase2_kd_loss(
                batch_items, comp_tok, comp_model, memory_bridge,
                dec_tok, dec_model, cfg, device, Hd,
            )
            if kd_loss is None or not torch.isfinite(kd_loss):
                continue
            kd_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step()

            if (kd_step + 1) % 100 == 0:
                print(
                    f"  [Phase-2] step {kd_step+1}/{cfg.qa_kd_steps_per_epoch}"
                    f" | kd_loss {kd_loss.item():.4f}"
                )

        ep_time = time.time() - ep_start
        print(
            f"[Epoch {ep+1}] avg loss: {sum(losses_log)/len(losses_log):.4f}"
            f" | time: {ep_time/60:.1f} min"
        )

    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time/3600:.2f} h")

    # ── Training loss curve ───────────────────────────────────────────────────
    if losses_log:
        steps = np.arange(1, len(losses_log) + 1)
        win = max(5, int(0.02 * len(losses_log)))
        smooth = np.convolve(losses_log, np.ones(win) / win, mode="valid")
        plt.figure(figsize=(8, 4))
        plt.plot(steps, losses_log, alpha=0.35, linewidth=1, label="per-step loss")
        plt.plot(steps[win - 1:], smooth, linewidth=2, label=f"moving avg (w={win})")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("CMC — Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig("training_loss.png", dpi=150)
        print("Loss curve saved to training_loss.png")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    save_checkpoint(cfg.checkpoint_dir, comp_model, comp_tok, memory_bridge, cfg)

    # ── Evaluation ───────────────────────────────────────────────────────────
    print("\nEvaluating on validation set …")
    results = evaluate_qa(
        val_examples, comp_tok, comp_model, memory_bridge,
        dec_tok, dec_model, cfg, device, Hd,
        limit=cfg.val_samples, print_every=100,
    )
    print("\n=== Evaluation Results ===")
    print(json.dumps(results, indent=2))

    # ── Efficiency report ─────────────────────────────────────────────────────
    if val_examples:
        ex = val_examples[0]
        wctx, _ = gold_char_window(ex["context"], ex["answer_start"], cfg.local_window_chars)
        local_ctx = build_local_window_tokens(
            ex["context"], ex["answer_start"], cfg.local_window_tokens, dec_tok
        )
        from cmc.compression import build_memory_vectors
        with torch.no_grad():
            mem = build_memory_vectors(
                ex["context"], comp_tok, comp_model, memory_bridge,
                cfg, device, Hd, training=False,
            )
        Mb = int(mem.size(0))

        report = kv_cache_report(
            n_layers=n_layers, n_heads=n_heads, head_dim=head_dim, d_model=Hd,
            prompt_len_baseline=len(dec_tok(wctx)["input_ids"]) + len(dec_tok(ex["question"])["input_ids"]),
            prompt_len_cmc=len(dec_tok(local_ctx)["input_ids"]) + len(dec_tok(ex["question"])["input_ids"]),
            mem_tokens=Mb, cfg=cfg,
        )
        print("\n=== Efficiency Report ===")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
