"""
models.py
---------
Neural network modules for the CMC framework.

MemoryBridge
    Norm-calibrated two-layer MLP that projects ContextEncoder hidden
    states (dimension H_c) into the frozen DecoderLLM embedding space
    (dimension H_d).  The output LayerNorm + tanh scaling ensures that
    projected CME vectors stay within the decoder's native embedding
    norm distribution.
"""

import torch
import torch.nn as nn


class MemoryBridge(nn.Module):
    """
    Cross-architecture projection from encoder space to decoder space.

    Architecture
    ------------
    LayerNorm(H_c) → Linear(H_c → H) → GELU → Linear(H → H_d)
      → LayerNorm(H_d) → tanh-norm-cap

    Parameters
    ----------
    din : int
        Hidden dimension of the ContextEncoder (H_c).
    dout : int
        Hidden dimension of the DecoderLLM (H_d).
    target_norm : float
        Mean ℓ₂ norm of the frozen decoder's input embedding matrix,
        computed once at initialisation.  Used as the norm cap ν̄.
    """

    def __init__(self, din: int, dout: int, target_norm: float) -> None:
        super().__init__()
        h = max(din, dout)
        self.proj = nn.Sequential(
            nn.LayerNorm(din, eps=1e-5),
            nn.Linear(din, h),
            nn.GELU(),
            nn.Linear(h, dout),
        )
        self.out_ln = nn.LayerNorm(dout, eps=1e-5)
        # Learnable scale s; norm cap ν = s · ν̄
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer(
            "target_norm", torch.tensor(float(target_norm)), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (*, H_c)

        Returns
        -------
        Tensor of shape (*, H_d), norm-capped to 2 · target_norm.
        """
        y = self.proj(x.float())
        # Numerical safety clamp
        y = torch.nan_to_num(y, nan=0.0, posinf=1e3, neginf=-1e3).clamp_(-6.0, 6.0)
        y = self.out_ln(y)
        # Norm-calibrated tanh: ỹ = ν · tanh(LayerNorm(ŷ) / ν)
        nu = self.scale.abs() * self.target_norm
        y = torch.tanh(y) * nu
        # Hard norm cap at 2 · target_norm
        norms = y.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        y = y * torch.minimum(torch.ones_like(norms), (2.0 * self.target_norm) / norms)
        return y
