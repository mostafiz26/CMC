"""
CMC — Context-to-Memory Compression framework.
"""

from .config import CMCConfig
from .models import MemoryBridge
from .compression import build_memory_vectors, ig_prune_token_ids, crop_past_with_memory
from .training import phase1_ae_ar_losses, phase2_kd_loss
from .evaluation import evaluate_qa, em_f1
from .utils import save_checkpoint, load_checkpoint, collect_examples
from .efficiency import kv_cache_report, estimate_total_attn_macs

__all__ = [
    "CMCConfig",
    "MemoryBridge",
    "build_memory_vectors",
    "ig_prune_token_ids",
    "crop_past_with_memory",
    "phase1_ae_ar_losses",
    "phase2_kd_loss",
    "evaluate_qa",
    "em_f1",
    "save_checkpoint",
    "load_checkpoint",
    "collect_examples",
    "kv_cache_report",
    "estimate_total_attn_macs",
]
