# LOOM — Implementation Plan

Repo root `PLAN.md`. Every agent reads this file plus `contracts.py`. Architecture diagram: `LOOM_proposal.html`.

**Objective:** build the model, run LIBERO, get the first score. Then RoboTwin as the decision gate, then pretraining.

**First milestone: an end-to-end LIBERO success rate.** Everything in Phase 1A exists to reach that number. Everything else is Phase 1B and runs concurrently but must not block it.

---

## 1. Method

A persistent belief state, transformed by operators. The action *is* the operator; the body-specific decoder only realizes it.

```
BELIEF        z_t     = E(o_t, ℓ, z_{t−1})                z ∈ ℝ^(K×D),  K=128, D=768

OPERATOR      c       = q_Δ(z_t, z_{t+8})     action-free
                      = q_a^e(a_{t:t+7}, z_t) action-labelled
                        c ∈ Δ^(M−1), ‖c‖₀ ≤ 4,  M=128

DYNAMICS      ẑ_{t+8} = A(c) z_t + b(c),      A(c) = Σ_m c_m A_m
                        ‖A(c)‖₂ ≤ ρ,  ‖b(c)‖ ≤ B_max      (both by convexity)

REALIZE       â       = D_e(p_t, c) → (H_OP, dof_e)       8 control steps
                        p_t = proprio at t. NOT z: see 4.C.

POLICY        c       ~ π_c(c | z_t, ℓ)                   inference path

PLAN          ĉ_{1:4} ~ π_c,  N=1000 samples
              score   = Φ(ẑ_4, ℓ)
              execute D_e(z_t, ĉ_1*)                      root segment only
```

Each `A_m` is block-diagonal in real 2×2 rotation-decay blocks. A block `r·[[cos ω, −sin ω],[sin ω, cos ω]]` is the matrix form of `r·e^{iω}`, and those are closed under addition — so a convex mixture of blocks is again one block, and both bounds follow from the triangle inequality.

Two losses carry the representation. `L_dyn` teaches what a transformation does; `L_act` teaches how this body produces it. `L_proposal` is behaviour cloning of `π_c` and is not optional — without it there is no way to obtain `c` at inference, because `q_Δ` needs the future and `q_a` needs the ground-truth action.

**Filtering and prediction are separate.** `E` is nonlinear, unconstrained, and runs once per executed segment (3.75 Hz). The rollout is pure affine algebra and runs `N × DEPTH` times per cycle. All expressivity lives in the filter; none of it is inside the planning loop.

---

## 2. Contracts

Write these two files first, then freeze. Nothing else starts until they land.

```python
# contracts.py — FROZEN
from typing import Protocol, TypedDict
from torch import Tensor

# ── temporal ──────────────────────────────────────────────────────────
FPS_CANONICAL = 30      # every dataset resampled to this before segmenting
H_OP          = 8       # control steps per operator → 267 ms
DEPTH         = 4       # planning horizon, in operators
H_PLAN        = H_OP * DEPTH        # 32 canonical steps, 1.07 s
N_STATES      = DEPTH + 1           # 5 operator-boundary states per window

# ── model ─────────────────────────────────────────────────────────────
K       = 128           # belief slots
D       = 768           # slot width, MUST be even
M       = 128           # operator bank size
TOPK    = 4             # nonzero coefficients
RHO     = 0.98          # spectral radius bound per operator
B_MAX   = 1.0           # norm bound per bias
EMA_TAU = 0.996

# ── tensor aliases (documentation only) ───────────────────────────────
# Belief        (B, K, D)              real
# Coeff         (B, M)                 simplex, <= TOPK nonzero
# LamPair       (B, K, D//2) x2        real (a, b) meaning a + ib, |a+ib| <= RHO
# ActionSegment (B, H_OP, dof_e)       ONE operator's worth. NEVER H_PLAN.

class ObsFeats(TypedDict):
    views:   Tensor   # (B, V, P, F)  V streams; tactile gel-pads are just views
    proprio: Tensor   # (B, dof_e)
    lang:    Tensor   # (B, L, F)

class TransitionWindow(TypedDict):
    feats:      list[ObsFeats]   # length N_STATES = 5, at canonical frames 0/8/16/24/32
    actions:    Tensor | None    # (B, DEPTH, H_OP, dof_e); None for action-free data
    lang:       Tensor
    embodiment: str              # HOMOGENEOUS within a batch
    src_fps:    float            # original rate, needed to invert resampling at eval

class Estimator(Protocol):                                # shared
    def forward(self, feats: ObsFeats, z_prev: Tensor | None) -> Tensor: ...

class Bank(Protocol):                                     # shared
    def mix(self, c: Tensor) -> tuple[Tensor, Tensor]: ...      # Coeff -> (a, b)
    def bias(self, c: Tensor) -> Tensor: ...                    # Coeff -> (B,K,D)
    def step(self, c: Tensor, z: Tensor) -> Tensor: ...         # ONE affine step
    def rollout(self, c_seq: Tensor, z: Tensor) -> Tensor: ...
    # (B,N,DEPTH,M), (B,K,D) -> (B,N,K,D). Sequential over DEPTH.
    # There is NO compose(). Composing affine maps gives (A2A1, A2b1+b2);
    # multiplying lambdas alone silently discards the bias.

class QDelta(Protocol):                                   # shared
    def forward(self, z_t: Tensor, z_next: Tensor) -> Tensor: ...

class QAction(Protocol):                                  # ONE PER EMBODIMENT
    def forward(self, a_seg: Tensor, z: Tensor) -> Tensor: ...  # a_seg (B,H_OP,dof_e)

class Decoder(Protocol):                                  # ONE PER EMBODIMENT
    def forward(self, proprio: Tensor, c: Tensor) -> Tensor: ...   # -> (B,H_OP,dof_e)
    def loss(self, proprio: Tensor, c: Tensor, a_seg: Tensor) -> Tensor: ...

class Proposal(Protocol):                                 # shared
    def sample(self, z: Tensor, lang: Tensor, n: int) -> Tensor: ...   # -> (B,n,M)
    def log_prob(self, z: Tensor, lang: Tensor, c: Tensor) -> Tensor: ...

class Potential(Protocol):                                # shared, R3 only
    def forward(self, z: Tensor, lang: Tensor) -> Tensor: ...          # -> (B,)

class Policy(Protocol):
    """The ONLY interface eval depends on."""
    def reset(self) -> None: ...
    def act(self, obs: dict, instruction: str) -> "np.ndarray": ...
```

`stubs.py` gives every Protocol a shape-correct random implementation. That is what unblocks all six teams on day one.

**Done when** `pytest tests/test_contracts.py` passes: `H_OP * DEPTH == H_PLAN`, `D % 2 == 0`, `len(feats) == N_STATES`, every stub returns documented shapes.

### Parameter budget

| module | scope | params |
|---|---|---|
| vision + text encoders | frozen | — |
| `E` estimator | shared | 150 M |
| operator bank | shared | 25 M |
| `q_Δ` | shared | 30 M |
| `π_c` proposal | shared | 50 M |
| `Φ` potential | shared | 0.2 M |
| `q_a^e` | per body | 30 M |
| `D_e` | per body | 20 M |

**≈ 255 M shared + 50 M per embodiment.** Do not inflate. A 10-block Perceiver at `d=768` is ~150M; if your estimator is coming out at 600M you have over-specified the FFNs.

---

## 3. File ownership

An agent may only create or edit files in its own row. Cross-team changes go through the integration owner. Each team works in its own git worktree on `team-<x>`.

| Team | Owns |
|---|---|
| — | `contracts.py`, `stubs.py`, `PLAN.md` |
| **A** data | `loom/data/**`, `tests/test_data.py` |
| **B** core | `loom/model/**`, `tests/test_model.py` |
| **C** heads | `loom/heads/{q_delta,q_action,decoder}.py`, `loom/losses/**`, `tests/test_{heads,losses}.py` |
| **D** train | `loom/train/**`, `configs/**`, `tests/test_train.py` |
| **E** policy | `loom/heads/{proposal,potential}.py`, `loom/search/**`, `tests/test_search.py` |
| **F** eval | `loom/eval/**`, `tests/test_eval.py` |

```
loom/
├── contracts.py                FROZEN
├── stubs.py                    FROZEN
├── data/    adapters/{libero,robotwin,openneo,agibot,robomind,oxe,ego4d}.py
│            canonical.py  cache.py  loader.py
├── model/   estimator.py  bank.py  rollout.py
├── heads/   q_delta.py  q_action.py  decoder.py  proposal.py  potential.py
├── losses/  dyn.py  act.py  proposal_bc.py  balance.py
├── search/  shooting.py
├── train/   loop.py  fsdp.py  schedule.py  slurm/{r0a,r0b,r1,r2,r3}.sbatch
├── eval/    libero.py  robotwin.py  libero_plus.py  runner.py
configs/     base.yaml  r0a.yaml  r0b.yaml  r1.yaml  r2.yaml  r3.yaml
tests/
```

---

## 4. Phase 1A — critical path to the first LIBERO score

Everything here is required for R0-A. Nothing else is. Teams do 1A items before touching 1B.

### A · LIBERO adapter + loader

> `adapters/libero.py`, `canonical.py`, `cache.py`, `loader.py`.
>
> LIBERO only for now — do not build seven adapters before the first score exists.
>
> **`canonical.py`** resamples any trajectory to `FPS_CANONICAL = 30` and segments into `TransitionWindow`: five states at canonical frames 0/8/16/24/32 plus four 8-step action segments. Record `src_fps` in the window — eval needs it to invert the resampling. Different datasets ship at different control rates, and 8 steps must mean the same physical duration everywhere or the shared bank is meaningless.
>
> **`cache.py`** encodes once with the frozen vision tower; never re-encode. **Storage format is a day-one profiling call:** fp16 patch tokens run ~1 MiB per 2-stream LIBERO state and ~3 MiB per 7-stream state, so a 5-state window is 5–15 MiB. Options are fp16, int8-quantised, or keeping the tower in-graph under `no_grad` and paying FLOPs instead of I/O. Measure all three.
>
> **`loader.py`** — **every minibatch contains exactly one embodiment.** This fixes `dof`, proprio width, action normalisation, and head dispatch, and eliminates all action padding and masking. Mix embodiments between batches, never within. LIBERO is single-embodiment, so build the dispatch anyway and test it with two synthetic bodies.
>
> **Done when** `tests/test_data.py` passes: resampling verified against known-rate fixtures; windows have 5 states and 4 segments at the right offsets; `actions=None` path works; batches are embodiment-homogeneous; cache round-trips; **pipeline sustains ≥1.3× measured training consumption without GPU starvation.**

### B · estimator, bank, rollout

> **`estimator.py`** — Perceiver. `K=128` learned queries cross-attend to `concat(views, proprio, lang, z_prev)`. 10 blocks, `d=768`, 16 heads, pre-LN. **Learned slot embeddings, no RoPE on the slot axis** — the 128 queries are slots, not a sequence, and there is no reason slot 12 should be nearer slot 13 than slot 80. Spatial structure stays inside the frozen vision tokens.
>
> **`bank.py`** — the highest-leverage file. **Implement the 2×2 real algebra directly. Do not use `torch.view_as_complex`: PyTorch has no complex-bf16 dtype and the entire run is bf16.**
>
> ```python
> # params: log_r (M,K,D//2), omega (M,K,D//2), b_raw (M,K,D)
> r        = RHO * torch.sigmoid(self.log_r)
> A_a, A_b = r * torch.cos(self.omega), r * torch.sin(self.omega)
>
> n      = self.b_raw.flatten(1).norm(dim=1).clamp(min=B_MAX).view(M, 1, 1)
> A_bias = B_MAX * self.b_raw / n                      # ‖A_bias[m]‖ <= B_MAX
>
> def mix(self, c):
>     return (torch.einsum('bm,mkj->bkj', c, A_a),
>             torch.einsum('bm,mkj->bkj', c, A_b))
>
> def step(self, c, z):
>     a, b = self.mix(c)
>     zr   = z.reshape(*z.shape[:-1], D // 2, 2)
>     x, y = zr[..., 0], zr[..., 1]
>     out  = torch.stack([a * x - b * y, b * x + a * y], dim=-1)   # 4 real ops
>     return out.reshape(*z.shape) + self.bias(c)
> ```
>
> Both bounds are then free: `|Σ c_m λ_m| ≤ Σ c_m r_m ≤ ρ` and `‖Σ c_m b_m‖ ≤ Σ c_m ‖b_m‖ ≤ B_MAX`.
>
> **Initialisation matters.** A depth-4 rollout is four applications; `r` must init near `ρ` with a spread across the `D/2` axis so different channels carry different timescales. Use S4D-style `log_r` init, not a constant.
>
> **`rollout.py`** — sequential over `DEPTH`, batched over `N`:
> ```python
> z = z.unsqueeze(1).expand(B, N, K, D)
> for d in range(DEPTH):
>     z = bank.step(c_seq[:, :, d], z)
> ```
> **There is no `compose()`.** Affine composition is `(A₂A₁, A₂b₁+b₂)`; multiplying lambdas alone drops the bias.
>
> **Done when** `tests/test_model.py` passes: `‖A(c)‖₂ ≤ RHO` and `‖b(c)‖ ≤ B_MAX` over 10k random simplex draws; `rollout` matches a naive loop to 1e-5; **`N=1000, DEPTH=4` under 5 ms on one A100**; estimator ≥30 Hz with 7 streams; no dtype promotion out of bf16.

### C · q_Δ, q_a, decoder, L_dyn, L_act, L_proposal, L_balance

> **`q_delta.py`** — shared. `(z_t, z_{t+8}) → Coeff`. MLP on `concat(pool(z_t), pool(z_next), pool(z_next − z_t))`. Logits over `M`, then **hard top-4 straight-through, renormalised to sum to 1**. The renormalisation puts `c` on the simplex and is what makes Team B's bounds hold — a plain softmax breaks both.
>
> **`q_action.py`** — **`ModuleDict` keyed by embodiment.** `(a_seg (B,H_OP,dof_e), z) → Coeff`, same top-4 head. Trained by regression onto `sg(q_Δ(z_t, z_{t+8}))`, so both encoders write into one coefficient space by construction. No KL, no adversarial term, no separate alignment loss.
>
> **`decoder.py`** — **`ModuleDict` keyed by embodiment.** `D_e(proprio, c) → (B, H_OP, dof_e)`. **One operator = one 8-step segment. Never 32.** Conditional flow matching.
>
> **The belief is NOT an input** (owner-authorised contract change, after R0-A). Given the full `(K, D)` belief, predicting an 8-step segment is behaviour cloning, and behaviour cloning needs nothing from `c`: R0-A measured `act/decode` falling 0.2489 → 0.0559 while `c_a` held 2–3 distinct top-4 supports over 64 real training windows, i.e. `L_act` put no pressure on the coefficient at all. `proprio` is `ObsFeats["proprio"]`, `(B, dof_e)` — one timestep of the body's own state. It says where the arm is; it cannot say where the target is, so it cannot substitute for `c`.
>
> **`losses/dyn.py`**
> ```
> L_dyn = Σ_{h=1..DEPTH} w_h · (1 − cos(LN(ẑ_{t+8h}), sg(LN(z̄_{t+8h}))))
> w = {1.0, 0.5, 0.25, 0.125}
> ```
> `ẑ` from sequential `bank.step`; targets from the EMA estimator (`τ = 0.996`) under stop-grad.
>
> Expose `dyn.negatives ∈ {none, within_trajectory}`, default **`within_trajectory`**. `loom/train/loop.py` calls `dyn_loss` directly — it used to compute a bare `1 − cos(A(c)z, z⁺)` inline with no negatives at all, so every config's `negatives:` key was inert.
>
> Log **`Δ_sel = d(A(c_other) z, z⁺) − d(A(c_true) z, z⁺)`** with `c_other = c.roll(1, dims=0)`, i.e. a *real* coefficient from another window. This, and not `Δ_op`, is the discrimination test: `Δ_op` compares against a uniform random simplex point and only says the bank is alive. R0-A measured `Δ_sel` at +0.0002 (ctrl) / +0.0000 (zinit). Negatives are `c` from another segment of the **same trajectory, ≥2 segments away** — same scene, same body, genuinely different effect. Do **not** use uncurated in-batch negatives: two bodies producing the same world effect would become negatives for each other, which is precisely the opposite of what a shared operator bank should learn.
>
> Log every step: **`Δ_op = d(A(c_rand)z, z⁺) − d(A(c_true)z, z⁺)`, which must be > 0.** Mind the sign: for a distance, the true operator should be *closer*. Latent states 8 steps apart are ~0.95 cosine-similar before training, so `A(c) ≈ I` nearly satisfies `L_dyn` while `c` carries nothing. If `Δ_op` flatlines in the first few thousand steps, the model has collapsed to a plain latent policy — flip `negatives` to `within_trajectory` before burning the full run. This is a build assert, not a metric.
>
> **`losses/act.py`** — wrapper over `Decoder.loss`, dispatched by embodiment.
> **`losses/proposal_bc.py`** — `−log π_c(sg(c_a) | z, ℓ)`. Not loss creep: this is the only thing that makes the model executable.
> **`losses/balance.py`** — coefficient `contracts.BALANCE_COEF`. Hard top-k already provides sparsity; this only prevents dead operators.
>
> The executed form is the **Switch auxiliary** `M · Σ_m f_m P_m` (`loom/train/loop.py::_switch_balance`), with `f_m` the fraction of routing slots that went to `m` and `P_m` the mean **dense** router probability — floor 1.0, ceiling `M/TOPK = 32`. It replaced `KL(mean_batch(c) ‖ uniform(M))`, which is a function of `c` alone and therefore blind to how nearly an unselected operator was chosen; the coefficient went `3e-3 → 1e-2` with it. `losses/balance.py` still holds the KL as the reference.
>
> **Done when** `tests/test_losses.py` passes: `L_dyn` decreases on a synthetic task where `c` is the only informative input; `Δ_op > 0` on that task; top-4 output verified on-simplex; per-embodiment dispatch verified with two different `dof`.

### D · training loop, FSDP, R0-A config

> **`fsdp.py`** — FSDP full-shard on `E`; bank and heads replicated. **bf16 throughout — A100 has no FP8, do not add an FP8 path.** Activation checkpointing on estimator blocks. Frozen tower never enters the graph.
>
> **`loop.py`** — `python -m loom.train --config configs/rX.yaml`. Per-stage loss subsets, per-module LR groups, EMA maintenance, resume, deterministic seeding.
>
> **`schedule.py`** — AdamW, cosine, 2k warmup, grad clip 1.0. **Bank parameters at 10× lower LR than the estimator** — spectral parameters are sensitive.
>
> **`slurm/*.sbatch`** — R1/R2/R3 run 3–6 days and **will** be preempted. Each needs `--requeue`, automatic latest-checkpoint resume, and checkpointed optimizer + scheduler + EMA + sampler cursor + RNG state + a stable W&B run ID, plus a graceful save near walltime.
>
> **Done when** `tests/test_train.py` passes: a 50-step stub run completes; FSDP works on 2 GPUs; **survives an artificial `SIGTERM` and resumes with continuous loss** — do not require bit-identical float after a distributed restart; memory profile fits at the configured batch size.

### E · proposal head

> **`proposal.py`** — `π_c(c | z, ℓ)`, ~50 M. **Plackett–Luce over sampling without replacement**, fully specified so PPO/GRPO later have a real log-probability:
>
> ```
> network → logits ℓ ∈ ℝ^M
> sample k = TOPK indices S = (i₁…i_k) sequentially WITHOUT replacement
> weights deterministic given S:  c_{i_j} = softmax(ℓ restricted to S)
> log π(S) = Σ_{j=1..k} [ ℓ_{i_j} − logsumexp_{m ∉ {i₁…i_{j−1}}} ℓ_m ]
> ```
> The stochastic variable is the ordered subset.
>
> **This is on the critical path for the first score.** `π_c` is the inference path: at test time there is no ground-truth action and no future state, so neither `q_a` nor `q_Δ` is available. Without a BC-trained `π_c` the model cannot be evaluated at all.
>
> **Done when** `log_prob` matches brute-force enumeration at `M=6, k=2`, and gradients check against finite differences.

### F · LIBERO eval harness

> **`libero.py`, `runner.py`.** You depend on **nothing but `contracts.Policy`** — build against `stubs.Policy` from hour one. This workstream is historically the one that arrives late; it must not be.
>
> Inference path: `z ← E(o)`, `c ← argmax π_c(·|z,ℓ)`, `a ← D_e(z,c)`, execute the 8-step segment, re-filter. No search in R0.
>
> **Resample the decoded segment back to the environment's control rate.** The decoder emits `H_OP` steps at `FPS_CANONICAL = 30`; LIBERO's env does not run at 30 Hz. Use `src_fps` and interpolate. Getting this wrong produces a model that trains fine and scores near zero.
>
> **Replicate the evaluation protocol of the paper whose table we compare against (Light-WAM Table 1). Do not invent an episode count.** If the protocol is unstated, use 10 episodes/task × 10 tasks × 4 suites over 3 seeds and state it.
>
> `runner.py`: `python -m loom.eval --bench libero --ckpt <path> --out results.json`, parallel across GPUs, emitting markdown in the proposal's column order so numbers paste directly.
>
> **Done when** the harness runs end to end on `stubs.Policy` and emits a correctly-shaped table with random success rates. Plumbing is the deliverable; numbers come later.

---

## 5. Phase 1B — concurrent, must not block the first score

| Team | Item |
|---|---|
| A | Remaining adapters: RoboTwin, OpenNeoData, AgiBot, RoboMIND, OXE, Ego4D / Ego-Exo4D / HoloAssist |
| E | `potential.py` (`Φ`), `search/shooting.py` |
| F | `robotwin.py`, `libero_plus.py` |
| D | `configs/{r0b,r1,r2,r3}.yaml` |

### Search — batched shooting, no MCTS

```python
C     = proposal.sample(z, lang, n=N)     # (B, N, DEPTH, M)
leaf  = bank.rollout(C, z)                # (B, N, K, D)
score = potential(leaf, lang)             # (B, N)
best  = score.argmax(1)
return C[arange(B), best, 0]              # root segment only
```

Score is `Φ(ẑ_DEPTH, ℓ)` alone — the root term is identical across candidates and cancels in the argmax. No uncertainty or cost terms; they have no contract and inventing them reintroduces drift. Top-4 mixtures mean the action set is `C(128,4)` supports times continuous weights, so "128-arm MCTS" is not well-posed, and at depth 4 there is no prefix reuse worth a tree.

Realizability gate: reject a root `c` when `‖q_a(D_e(z,c), z) − c‖ > τ`; fall through to the runner-up.

---

## 6. Merge protocol

1. Merge only when your own tests are green **against stubs**.
2. Integration order **B → C → A → E → D → F**. Core first; everything shape-depends on it.
3. After each merge, swap one stub for the real module and re-run the suite. Breakage is owned by the merging team, not the integrator.
4. Nobody edits `contracts.py`. A genuine contract change halts Phase 1, is made once, and all six teams rebase.

**Phase 2 exit:** `train --config configs/r0a.yaml --steps 100` runs on real modules, and `eval --bench libero --ckpt <that>` emits a table.

---

## 7. Runs

| run | data | trains | losses | inference | GPUs | time |
|---|---|---|---|---|---|---|
| **R0-A** | LIBERO, from scratch | `E`, bank, `q_Δ`, `q_a`, `D_e`, `π_c` | dyn + act + proposal + balance | `π_c → D_e` | 16 | **~8 h** |
| **R0-B** | RoboTwin 2.0, from scratch | same | same | `π_c → D_e` | 16 | ~1 d |
| **R1** | Ego4D, Ego-Exo4D, HoloAssist | `E`, bank, `q_Δ` | dyn + balance | — | 64 | ~4 d |
| **R2** | OpenNeoData + AgiBot + RoboMIND + OXE | + `q_a`, `D_e`, `π_c` | dyn + act + proposal + balance | `π_c → D_e` | 64 | ~6 d |
| **R3** | sim rollouts, init from R2 | `Φ`, `π_c` (RL); light `E`/bank tune | potential + GRPO | **shooting** | 32 | ~3 d |

**R2 warm-up: freeze `E` *and* the bank together for the first 30% of steps.** Freezing the bank alone does not preserve the coefficient space — the operators would be fixed vectors in a drifting basis. With both frozen, `q_a` and `D_e` are forced into the coordinate system R1 established. Unfreeze everything after. This is what lets you drop an explicit alignment loss.

### R0-A — the first score

LIBERO, from scratch, ~8 hours. This is the plumbing proof and the first number in the table.

Read it correctly: **a low score proves something is broken; a high score does not prove the method works.** A June 2026 audit (arXiv 2606.04233) found that on LIBERO a 0.09B probe with no language encoder and no robotics pretraining scores 97.6 — above π0.5 at 96.9 and Fast-WAM at 97.0 in our own baseline table — and that only 19.8% of LIBERO SOTA claims are provably significant. So R0-A tells you the pipeline is alive, not that the operator formulation is good.

If R0-A lands under ~85, stop and debug rather than proceeding. Check `Δ_op` first, then the action-rate resampling in the eval harness — those are the two failure modes that produce a trained model with a near-zero score.

### R0-B — the decision gate

RoboTwin 2.0 clean, from scratch. The same audit finds RoboTwin 2.0 and RoboCasa fail fewer diagnostics, so this is where a number actually discriminates.

| result | action |
|---|---|
| **< 55** | **Kill.** The operator formulation is not competitive as a policy class |
| **55 – 75** | Proceed to R1/R2. π0 from scratch is 65.9 clean |
| **≥ 75** | Strong pass |

Three days total across R0-A and R0-B before committing to the 13-day pretraining chain.

---

## 8. Results

Three tables. Baseline rows are already filled and **copied from a single source per table** — do not re-run baselines, and never assemble a table across papers. Fast-WAM's LIBERO average appears in print as 97.6, 97.0, and 97.60 in three different papers.

**Standard LIBERO** (source: Light-WAM Table 1) — fill `LOOM · R0-A` and `LOOM · R2`.

| method | params | emb. PT | spatial | object | goal | long | avg |
|---|---|---|---|---|---|---|---|
| Diffusion Policy | — | ✗ | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| OpenVLA | 7 B | ✓ | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 | 3 B | ✓ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| VLA-Adapter | 0.6 B | ✗ | 96.0 | 96.8 | 97.4 | 94.4 | 96.2 |
| π0.5 | 3 B | ✓ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| Fast-WAM | 6 B | ✗ | 97.0 | 99.4 | 96.6 | 94.8 | 97.0 |
| Motus | 8 B | ✓ | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| LingBot-VA | 5.3 B | ✓ | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| **LOOM · R0-A** (belief decoder, no hinge, KL balance) | 0.3 B | ✗ | 5.3 | 15.0 | 13.3 | 0.0 | 8.4 |
| **LOOM · R0-A** (belief decoder, no hinge, KL balance, learned z₀) | 0.3 B | ✗ | 15.7 | 15.0 | 19.7 | 1.0 | 12.8 |
| **LOOM · R0-A** · ctrl, three fixes | 0.3 B | ✗ | 0.7 | 8.0 | 0.0 | 0.0 | 2.2 |
| **LOOM · R0-A** · zinit, three fixes | 0.3 B | ✗ | 1.0 | 2.7 | 0.3 | 0.0 | 1.0 |
| **LOOM · R2** | 0.3 B | ✓ | | | | | |

**RoboTwin 2.0** (source: Fast-WAM Table 1 + per-task appendix, randomized column) — fill `R0-B`, `R2`, `R3`.

| method | clean | rand | hanging mug | turn switch | place can basket | handover block |
|---|---|---|---|---|---|---|
| π0 | 65.9 | 58.4 | — | — | — | — |
| π0.5 | 82.7 | 76.8 | 17 | 54 | 62 | 57 |
| Motus | 88.7 | 87.0 | 38 | 78 | 76 | 73 |
| Fast-WAM | 91.9 | 91.8 | 62 | 59 | 69 | 81 |
| LingBot-VA | 92.9 | 91.5 | 28 | 45 | 84 | 78 |
| **LOOM · R0-B** | | | | | | |
| **LOOM · R2** | | | | | | |
| **LOOM · R3** | | | | | | |

**LIBERO-Plus zero-shot transfer** (source: OA-WAM Table 2; Geo Avg = mean of camera/robot-init/layout) — fill `R2`, `R3`.

| method | camera | robot init | layout | geo avg | light | backgnd | language | noise | total |
|---|---|---|---|---|---|---|---|---|---|
| HoloBrain-0 | 65.5 | 58.2 | 79.5 | 67.7 | 88.1 | 90.3 | 78.7 | 66.9 | 74.0 |
| GE-Act | 60.7 | 77.0 | 80.2 | 72.6 | 95.8 | 86.0 | 77.4 | 90.9 | 80.3 |
| π0.5 | — | — | — | 79.5 | — | — | — | — | — |
| Cosmos-Policy | 75.8 | 63.3 | 82.2 | 73.8 | 96.5 | 88.9 | 81.7 | 92.7 | 82.2 |
| OA-WAM | 80.5 | 89.6 | 82.8 | 84.3 | 96.5 | 95.9 | 85.3 | 75.6 | 83.9 |
| **LOOM · R2** | | | | | | | | | |
| **LOOM · R3** | | | | | | | | | |

Light and background sit at ~96 and are solved. Camera and robot-init at 60–80 are where the headroom is.

**R0-A, all four rows: 1200 episodes each** (10 ep/task × 10 tasks × 4 suites × 3 seeds, max 512 env steps, real LIBERO, `--require-real --op-stats`). The three seeds are **replicates, not new scenes** — `LiberoEnv` selects its init state by `init_states[episode % 50]` and `set_init_state` overrides `env.seed()` — so there are **400 distinct conditions**, and the interval to quote is `sqrt(p(1−p)/400)`, not `/1200`. Measured replication: 94.5% (ctrl) / 97.0% (zinit) of conditions give a byte-identical step count across all three seeds, against 80.2% / 73.0% on the two rows above.

The "three fixes" rows are `D_e(proprio, c)`, the live `within_trajectory` hinge, and the Switch balance at `BALANCE_COEF = 1e-2`; git `d87f805`. They are **lower**, and the reason is not ambiguous: `Δ_sel`, the only test of whether `c` names *this* window's transition rather than *a* transition, sits at 7e-08 (ctrl) / −1e-07 (zinit) at step 7000, two orders of magnitude BELOW its value at initialisation (~1e-04). `π_c` then selects one operator for the entire benchmark — selection entropy 0.000 nats at every replan index, over 1200 episodes and 40 tasks, against the `ln 128 = 4.852` ceiling.

---

## 9. Standing rules

- bf16 only. A100 has no FP8. **No `view_as_complex` — there is no complex-bf16 dtype.**
- The frozen tower never enters the training graph. Features are cached.
- No pixel decoding, no VAE, no video DiT anywhere in this repo.
- `z` is real throughout. The 2×2 block algebra is four real elementwise ops.
- `c` lives on the simplex with hard top-4. Every bound depends on this.
- **One `c` = one operator = `H_OP` = 8 control steps.** Never `H_PLAN`.
- Multi-step rollout is sequential affine. **Never multiply lambdas alone.**
- All data resampled to 30 Hz before segmentation; all decoded actions resampled back to the environment rate before execution.
- Batches are embodiment-homogeneous. `q_a` and `D_e` are per-embodiment; `E`, bank, `q_Δ`, `π_c`, `Φ` are shared.
- Do not add losses beyond `dyn + act + proposal + balance`, plus `potential + RL` in R3.
- Do not add analysis, plots, ablation grids, or diagnostic studies. The deliverable is three success-rate tables.
