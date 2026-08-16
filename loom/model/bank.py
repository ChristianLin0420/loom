"""
LOOM — the operator bank.

`A(c) = sum_m c_m A_m`, `b(c) = sum_m c_m b_m`, with `c` on the simplex.

Each `A_m` is block-diagonal in real 2x2 rotation-decay blocks

        r * [[cos w, -sin w],
             [sin w,  cos w]]

which is the matrix form of `r * e^{iw}` acting on the pair `(x, y)` read as
`x + iy`. Those blocks are closed under addition, so a convex mixture of blocks
is again one block and both bounds fall straight out of the triangle inequality:

    |sum_m c_m lambda_m| <= sum_m c_m r_m <= RHO
    ||sum_m c_m b_m||    <= sum_m c_m ||b_m|| <= B_MAX

Hard rules this file obeys (PLAN.md 9):

* **No `torch.view_as_complex` anywhere.** PyTorch has no complex-bf16 dtype and
  the entire run is bf16. The block product is four real elementwise ops.
* `z` is real throughout; nothing promotes out of the parameter dtype.
* **There is no `compose()`.** Composing affine maps gives `(A2 A1, A2 b1 + b2)`;
  multiplying lambdas alone silently discards the accumulated bias. Multi-step
  rollout is sequential (see `loom/model/rollout.py`).

Parameter budget: `log_r` and `omega` are `(M, K, D//2)` and `b_raw` is
`(M, K, D)`, so `2 * M*K*D/2 + M*K*D = 2 * M*K*D` = 25.2 M parameters at
M=K=128, D=768 — the 25 M row of the budget table.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from contracts import B_MAX, D, K, M, RHO

from loom.model.rollout import rollout as _rollout

__all__ = ["OperatorBank"]


# ═══════════════════════════════════════════════════════════════════════════
#  INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════
#
#  Defaults for the S4D-style spectral init. Time constants are measured in
#  *operator steps*; one operator is H_OP = 8 canonical frames = 267 ms, so
#  TAU_MIN = 1 is 0.27 s and TAU_MAX = 40 is 10.7 s. That brackets the range of
#  manipulation timescales we care about: a single contact event at the fast end,
#  a whole multi-stage task at the slow end.
#
#  TAU_MAX is bounded above by RHO: r = exp(-1/tau) must stay below RHO = 0.98,
#  i.e. tau < -1/log(RHO) = 49.5. 40 leaves the sigmoid comfortably unsaturated
#  (sigmoid(log_r) ~ 0.995, not 1 - 1e-9), which is what keeps gradients alive on
#  the slow channels.

TAU_MIN = 1.0
TAU_MAX = 40.0

#: std of the Gaussian jitter added to log_r, in logit units
LOG_R_JITTER = 0.1

#: Frobenius norm each operator's bias is initialised to, as a fraction of B_MAX
BIAS_INIT_NORM = 0.1


class OperatorBank(nn.Module):
    """`M` block-diagonal rotation-decay operators plus `M` bounded biases.

    Satisfies `contracts.Bank`. Drop-in replacement for `stubs.StubBank`.

    Shapes
    ------
    `mix(c)`  : `(..., M)` -> two `(..., K, D//2)` tensors `(a, b)`, meaning
                `a + ib` per block, with `sqrt(a^2 + b^2) <= RHO`.
    `bias(c)` : `(..., M)` -> `(..., K, D)`, with Frobenius norm `<= B_MAX`.
    `step`    : `(..., M)`, `(..., K, D)` -> `(..., K, D)`.
    `rollout` : `(B, N, DEPTH, M)`, `(B, K, D)` -> `(B, N, K, D)`.

    `mix`, `bias` and `step` broadcast over arbitrary leading dimensions: `c` may
    be `(B, M)` against `z` `(B, K, D)`, or `(B, N, M)` against `z`
    `(B, N, K, D)`. This is why the einsums are written `'...m,mkj->...kj'`.

    Initialisation
    --------------
    See `_init_spectral`. Short version: S4D-Lin, adapted to the bounded
    parameterisation `r = RHO * sigmoid(log_r)`.
    """

    def __init__(
        self,
        m: int = M,
        k: int = K,
        d: int = D,
        rho: float = RHO,
        b_max: float = B_MAX,
        tau_min: float = TAU_MIN,
        tau_max: float = TAU_MAX,
        log_r_jitter: float = LOG_R_JITTER,
        bias_init_norm: float = BIAS_INIT_NORM,
    ) -> None:
        super().__init__()
        if d % 2 != 0:
            raise ValueError(f"D must be even (2x2 blocks), got {d}")
        if not 0.0 < rho < 1.0:
            raise ValueError(f"rho must be in (0, 1), got {rho}")
        if not 0.0 < tau_min <= tau_max:
            raise ValueError(f"need 0 < tau_min <= tau_max, got {tau_min}, {tau_max}")
        if math.exp(-1.0 / tau_max) >= rho:
            raise ValueError(
                f"tau_max={tau_max} implies r=exp(-1/tau_max)={math.exp(-1.0 / tau_max):.4f} "
                f">= rho={rho}; the bounded parameterisation cannot reach it. "
                f"tau_max must be below {-1.0 / math.log(rho):.2f}."
            )

        self.m, self.k, self.d = m, k, d
        self.n_blocks = d // 2
        self.rho, self.b_max = float(rho), float(b_max)

        self.log_r = nn.Parameter(torch.empty(m, k, self.n_blocks))
        self.omega = nn.Parameter(torch.empty(m, k, self.n_blocks))
        self.b_raw = nn.Parameter(torch.empty(m, k, d))

        self._init_spectral(tau_min, tau_max, log_r_jitter)
        self._init_bias(bias_init_norm)

    # ── initialisation ────────────────────────────────────────────────────

    @torch.no_grad()
    def _init_spectral(self, tau_min: float, tau_max: float, jitter: float) -> None:
        """S4D-Lin, mapped through the bounded parameterisation.

        Why an init scheme at all, and why this one.

        A depth-4 rollout is four applications of `A(c)`. Whatever `r` is at init
        gets raised to the fourth power before `L_dyn` sees the leaf, so init sets
        the effective memory of the whole planner. Two failure modes bracket the
        choice:

        * `r` constant and small: the leaf is `b`-dominated, `z_0` has been
          forgotten, and `L_dyn` cannot attribute anything to `c`.
        * `r` constant and equal to RHO: every channel has the *same* timescale,
          the bank has one degree of freedom where it should have `D/2`, and the
          only thing distinguishing operators is the rotation angle.

        S4D fixes this by giving each channel its own pole. We use the S4D-Lin
        variant (linear frequency spacing) with the diagonal-decay spread that
        S4D's log-spaced-`dt` initialisation produces, expressed directly on the
        discrete pole instead of on a continuous `A` plus a step size, because our
        operator is already discrete (one application = one 8-frame segment).

        Concretely, for block channel `j` of `D/2`:

            tau_j = tau_min * (tau_max / tau_min) ** (j / (D/2 - 1))   log-uniform
            r_j   = exp(-1 / tau_j)                                    decay per step
            w_j   = pi * (D/2 - 1 - j) / (D/2 - 1)                     S4D-Lin, reversed

        * **Log-uniform `tau`** rather than uniform `r`: what matters downstream is
          the *ratio* of timescales, and a uniform grid on `r` puts almost all its
          resolution on fast modes. Log-uniform gives a 40x span of memory lengths
          with every octave equally represented, so the 90th/10th percentile ratio
          of `tau` is ~19x rather than ~1x.
        * **Reversed frequency ramp**: `w` descends from pi to 0 as `tau` ascends,
          so slow channels are low-frequency (long-lived, near-DC integrators) and
          fast channels are high-frequency. The alternative pairing wastes
          capacity: a mode that decays in one step cannot resolve a slow
          oscillation, and a mode that survives 40 steps oscillating at Nyquist
          aliases away.
        * `r` is stored as `log_r` with `r = RHO * sigmoid(log_r)`, so the bound is
          structural rather than a projection step. `tau_max = 40` maps to
          `sigmoid(log_r) ~ 0.995`: near RHO, but far enough off the rail that the
          slow channels still receive gradient.

        Symmetry between the `M` operators is broken by a small Gaussian jitter on
        `log_r`, a one-spacing jitter on `omega`, and a random rotation sign per
        element (`+w` and `-w` are genuinely different real 2x2 blocks — one is the
        transpose of the other). Without the jitter every operator would start as
        the same matrix and `L_balance` would be fighting an exactly degenerate
        bank.
        """
        n = self.n_blocks
        idx = torch.arange(n, dtype=torch.float64)
        span = idx / max(n - 1, 1)

        tau = tau_min * (tau_max / tau_min) ** span                 # (n,) ascending
        r0 = torch.exp(-1.0 / tau)                                  # (n,) ascending
        sig = (r0 / self.rho).clamp(max=1.0 - 1e-6)
        log_r0 = torch.log(sig) - torch.log1p(-sig)                 # logit

        omega0 = math.pi * (1.0 - span)                             # (n,) pi -> 0

        shape = (self.m, self.k, n)
        self.log_r.copy_(
            (log_r0.expand(shape).clone() + jitter * torch.randn(shape, dtype=torch.float64))
            .to(self.log_r.dtype)
        )

        w_jitter = math.pi / max(n - 1, 1)                          # one spacing
        sign = torch.where(torch.rand(shape) < 0.5, -1.0, 1.0).double()
        self.omega.copy_(
            (sign * (omega0.expand(shape).clone() + w_jitter * torch.randn(shape, dtype=torch.float64)))
            .to(self.omega.dtype)
        )

    @torch.no_grad()
    def _init_bias(self, norm: float) -> None:
        """Init each `b_m` well below `B_MAX`.

        `A_bias = B_MAX * b_raw / max(||b_raw||, B_MAX)` is the identity while
        `||b_raw|| < B_MAX`, so starting at `norm * B_MAX` leaves the *magnitude*
        of the bias learnable, not only its direction. Initialising at the cap
        (which `randn * 0.1` would do — `||randn(K,D)|| * 0.1 ~ 31`) pins every
        operator to the sphere `||b|| = B_MAX` and makes the radial gradient
        component exactly zero.
        """
        self.b_raw.normal_(0.0, norm * self.b_max / math.sqrt(self.k * self.d))

    # ── bank tensors ──────────────────────────────────────────────────────

    def lam_bank(self) -> tuple[Tensor, Tensor]:
        """Per-operator block eigenvalues as a real pair. `(M, K, D//2)` x2.

        `sqrt(a^2 + b^2) = r <= RHO` elementwise, by construction.
        """
        r = self.rho * torch.sigmoid(self.log_r)
        return r * torch.cos(self.omega), r * torch.sin(self.omega)

    def bias_bank(self) -> Tensor:
        """Per-operator bias, norm-capped at `B_MAX`. `(M, K, D)`."""
        n = self.b_raw.flatten(1).norm(dim=1).clamp(min=self.b_max).view(self.m, 1, 1)
        return self.b_max * self.b_raw / n

    # ── contracts.Bank ────────────────────────────────────────────────────
    #
    #  Why a dense contraction over all M=128 and not a top-4 gather.
    #
    #  `c` has at most TOPK=4 nonzeros, so `A(c)` is a 4-term weighted sum and the
    #  dense einsum does 32x more arithmetic than the math requires. It is still
    #  the right kernel, and the peak-memory argument for the gather does not
    #  hold. Per `mix` call at N=1000, bf16:
    #
    #      dense    12.58 GFLOP, reads the bank once (12.6 MB), writes 98.3 MB
    #      gather    0.39 GFLOP, reads 4 slices per candidate (393.2 MB), writes 98.3 MB
    #
    #  The gather trades 12.2 GFLOP of tensor-core work for 380 MB of extra,
    #  *strided* HBM traffic. On an A100 that is 0.10 ms at 125 TFLOPS bf16
    #  against 0.26 ms at 1.5 TB/s, before the index_select's access pattern
    #  costs anything. Measured on this box (CPU, fp32, N=256, 8 threads) the two
    #  agree numerically to 1e-5 and dense is 13x faster: 540 ms vs 7119 ms.
    #
    #  Peak memory is the same either way, because both schemes materialise the
    #  same mixed `(N, K, D//2)` tensor and *that* is what costs, not the bank.
    #  One rollout step at N=1000 holds ~790 MB of transient bf16 activations
    #  (a, b, the stacked product, the mixed bias, z); the DEPTH steps are
    #  sequential so the peak does not compound. This is inference-only memory —
    #  PLAN.md 5 runs the rollout inside the search with no autograd tape, and
    #  L_dyn rolls out at N=1.
    #
    #  If a larger N ever makes that uncomfortable, the lever is to chunk the N
    #  axis in `rollout` — peak falls linearly in the chunk size, total work is
    #  unchanged, and L2 reuse of the bank improves. Not a gather.

    def mix(self, c: Tensor) -> tuple[Tensor, Tensor]:
        """`(..., M)` on the simplex -> `(a, b)`, each `(..., K, D//2)`."""
        a_bank, b_bank = self.lam_bank()
        c = c.to(a_bank.dtype)
        return (torch.einsum('...m,mkj->...kj', c, a_bank),
                torch.einsum('...m,mkj->...kj', c, b_bank))

    def bias(self, c: Tensor) -> Tensor:
        """`(..., M)` on the simplex -> `(..., K, D)` with norm `<= B_MAX`."""
        bank = self.bias_bank()
        return torch.einsum('...m,mkd->...kd', c.to(bank.dtype), bank)

    def step(self, c: Tensor, z: Tensor) -> Tensor:
        """ONE affine step: `A(c) z + b(c)`.

        The 2x2 block product written out as four real elementwise ops. `z` is
        reshaped to `(..., D//2, 2)` so that adjacent channels pair up; nothing is
        ever viewed as complex.
        """
        a, b = self.mix(c)
        if a.ndim != z.ndim:
            raise ValueError(
                f"step: c and z must carry the same leading dims — got c {tuple(c.shape)} "
                f"against z {tuple(z.shape)}. Right-aligned broadcasting between "
                f"(B, K, D//2) and (B, N, K, D//2) would silently misalign the batch."
            )
        zr = z.reshape(*z.shape[:-1], self.n_blocks, 2)
        x, y = zr[..., 0], zr[..., 1]
        out = torch.stack([a * x - b * y, b * x + a * y], dim=-1)   # 4 real ops
        # `out.shape`, not `z.shape`: z may be a size-1 broadcast against c.
        return out.reshape(*out.shape[:-2], self.d) + self.bias(c)

    def rollout(self, c_seq: Tensor, z: Tensor) -> Tensor:
        """`(B, N, DEPTH, M)`, `(B, K, D)` -> `(B, N, K, D)`. Sequential over DEPTH."""
        return _rollout(self, c_seq, z)

    # nn.Module courtesy alias; `step` is the contract entry point.
    def forward(self, c: Tensor, z: Tensor) -> Tensor:
        return self.step(c, z)

    def extra_repr(self) -> str:
        return f"M={self.m}, K={self.k}, D={self.d}, rho={self.rho}, B_MAX={self.b_max}"
