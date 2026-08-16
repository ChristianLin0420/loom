"""
LOOM — `Phi`, the potential.  Team E.  R3 only (PLAN 5, Phase 1B).

`Phi(z_hat_DEPTH, lang) -> scalar per candidate` is the *entire* shooting score.
There is no root term (identical across candidates, cancels in the argmax), no
uncertainty term and no cost term — those have no contract and inventing them
reintroduces drift (PLAN 5, PLAN 9).

Budget: **0.2M parameters.**  This runs `N = 1000` times per planning cycle,
once per candidate leaf, so it must be a rounding error next to the 48M-param
proposal.  Pool `z` over slots, pool `lang` over tokens, small MLP on the pair.
Nothing else.  Any depth here buys nothing: the leaf belief is already the
output of a 150M-param filter pushed through four affine operators.

Shapes (`contracts.Potential`, and matching `stubs.StubPotential`):

    z (B, K, D)     -> (B,)
    z (B, N, K, D)  -> (B, N)

generally `(..., K, D) -> (...)`.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from contracts import D

__all__ = ["Potential"]


class Potential(nn.Module):
    """Tiny scalar potential on (belief, language).  ~0.22M params.

    `lang_dim` is a constructor argument (default 1152, SigLIP-so400m) because
    Team A has not finalised the text tower.
    """

    def __init__(self, dim: int = D, lang_dim: int = 1152, hidden: int = 96) -> None:
        super().__init__()
        self.dim, self.lang_dim, self.hidden = dim, lang_dim, hidden
        self.z_norm = nn.LayerNorm(dim)
        self.z_proj = nn.Linear(dim, hidden)
        self.lang_norm = nn.LayerNorm(lang_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, 1),
        )
        self.apply(self._init)

    @staticmethod
    def _init(mod: nn.Module) -> None:
        if isinstance(mod, nn.Linear):
            nn.init.trunc_normal_(mod.weight, std=0.02)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)

    def forward(self, z: Tensor, lang: Tensor) -> Tensor:
        """(..., K, D) x (B, L, F) -> (...).  `lang` may also be (B, F)."""
        if z.ndim < 3 or z.shape[-1] != self.dim:
            raise ValueError(f"belief must be (..., K, {self.dim}), got {tuple(z.shape)}")
        lead = z.shape[:-2]                                     # (B,) or (B, N)

        # pool first, normalise after: averaging K=128 slots shrinks the signal
        # by ~sqrt(K), and a LayerNorm before the mean would not undo that.
        zv = self.z_proj(self.z_norm(z.mean(dim=-2)))           # (..., hidden)

        if lang.ndim >= 3:
            lang = lang.mean(dim=-2)                            # (B, F)
        lv = self.lang_proj(self.lang_norm(lang))               # (B, hidden)
        # broadcast the per-batch language vector across the candidate axis
        lv = lv.reshape(lv.shape[0], *(1,) * (len(lead) - 1), self.hidden)
        lv = lv.expand(*lead, self.hidden)

        return self.head(torch.cat([zv, lv], dim=-1)).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return f"hidden={self.hidden}, params={self.n_params()/1e6:.3f}M"
