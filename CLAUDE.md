# LOOM

A persistent belief state, transformed by operators. The action *is* the operator; the
body-specific decoder only realizes it. Read `PLAN.md` first — it is the spec, the schedule,
and the file-ownership table. `LOOM_proposal.html` has the architecture diagram.

**Objective: build the model, run LIBERO, get the first score.** Then RoboTwin 2.0 as the
decision gate, then pretraining. The deliverable is three success-rate tables (PLAN §8).

## Non-negotiables

**`contracts.py` and `stubs.py` are FROZEN.** Nobody edits them. A genuine contract change
halts Phase 1, is made once, and every team rebases. `tests/test_contracts.py` is the gate.

**YOU MUST NOT use `torch.view_as_complex`.** There is no complex-bf16 dtype and the entire
run is bf16. The operator bank is four real elementwise ops on adjacent 2×2 pairs.

**YOU MUST NOT add a `compose()` to the bank.** Affine composition is `(A₂A₁, A₂b₁+b₂)`;
multiplying lambdas alone silently discards the bias. Rollout is sequential `step`.

**YOU MUST NOT add losses** beyond `dyn + act + proposal + balance`, plus `potential + RL`
in R3. No auxiliary alignment loss, no KL between `q_Δ` and `q_a`, no uncertainty or cost
terms in the search score.

**One `c` = one operator = `H_OP` = 8 control steps. Never `H_PLAN`.** A decoder that emits
32 steps trains fine and scores near zero.

**YOU MUST NOT report a result without evidence in the message.** Paste the test output, the
metric, or the W&B run id. "It works" and "this should fix it" are not results.

**Never `scancel` a training job.** `touch runs/<name>/STOP` — the sentinel exists so the job
checkpoints before exiting.

## Commands

```bash
.venv/bin/python -m pytest tests/test_contracts.py -q   # the Phase 0 gate, <10s
.venv/bin/python -m pytest -q                           # everything
bash scripts/submit.sh <run_name> <n_links>             # chain 4h links
touch runs/<run_name>/STOP                              # graceful stop; do NOT scancel
bash scripts/wandb_sync.sh <run_name>                   # from a LOGIN node, after each link
python -m loom.eval --bench libero --ckpt <path> --out results.json
```

## Environment — read before anything else

This cluster's driver is **CUDA 12.2**. `pip install torch` pulls a `+cu13x` wheel that
imports cleanly, reports `torch.cuda.is_available() == False`, and trains on CPU while
holding 8 A100s. Install torch explicitly:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu124 "torch==2.6.0"
```

- SLURM account is `edgeai_tao-ptm_image-foundation-model-clip` — **not** `edgeai`,
  which fails with "Invalid account or account/partition combination". Partitions
  `batch_block1/2/3` **do not exist**; the real 4 h list is
  `polar4,polar3,polar,grizzly`. QoS: `free,normal`.
- **Never list `batch_singlenode` on a `--nodes>1` job.** It carries QoS
  `128_nodes_per_user_1_node_per_job` (8 GPUs, 1970G per job). A partition list is
  *not* a fallback chain: sbatch validates the request against it and rejects the
  whole submission with `QOSMaxMemoryPerJob` rather than scheduling onto polar4.
  Every LOOM stage is multi-node, so this failed all five at submit time.
  `polar4/polar3/polar/grizzly` all carry `..._128_nodes_per_job`.
- Non-secret paths (`LOOM_DATA_ROOT`, `LOOM_CACHE_DIR`, `HF_HOME`, `TRITON_CACHE_DIR`)
  live in `scripts/env.sh`, sourced by **both** the sbatch and `logs/*_smoke.sh`. They
  were once duplicated in the smoke only, so a green 1-GPU smoke was followed by a
  16-GPU link that died at init on a missing cache. A gate that runs in a different
  environment than the thing it gates is not a gate.
- Hard **4 h** walltime cap on every partition, 8×A100-80GB per node. An "8 h" run in
  PLAN §7 means ≥2 chained links; "6 d" means ~40.
- **No container, no enroot, no `module load`.** A plain `.venv` activated in the sbatch.
- Compute nodes have **no outbound network**. `WANDB_MODE=offline` is set in every sbatch;
  sync from a login node with `bash scripts/wandb_sync.sh <run>`.
- The login node has **no GPU** and no `nvidia-smi`. Every correctness test must pass on
  CPU; perf and multi-GPU tests go behind `@pytest.mark.gpu` / `@pytest.mark.multigpu`.
- Credentials live in `.env.local` (gitignored), sourced by the sbatch. Never echo them.

## Code conventions

- Python 3.13, **bf16 only** (A100 has no fp8). FSDP full-shard on `E`; bank and heads
  replicated. Activation checkpointing must be `CheckpointImpl.NO_REENTRANT` — reentrant
  breaks FSDP.
- All schedules are step-based. Nothing may derive from wall clock or epoch count, or the
  LR curve changes across links.
- All randomness derives from `(seed, global_step, rank)`. No bare `torch.randn` in the
  training path.
- `z` is real throughout, `(B, K, D)`. `c` lives on the simplex with hard top-4 — every
  bound in the method depends on that renormalisation.
- Batches are **embodiment-homogeneous**. `q_a` and `D_e` are per-embodiment `ModuleDict`s;
  `E`, bank, `q_Δ`, `π_c`, `Φ` are shared.
- All data resampled to 30 Hz before segmentation; all decoded actions resampled back to the
  environment rate before execution.
- The frozen vision tower never enters the training graph. Features are cached.

## Gotchas that have bitten this cluster

- W&B step must never regress. Always log the `global_step` restored from checkpoint; W&B
  silently drops out-of-order rows.
- W&B offline prints `"resume will be ignored"`. Cosmetic — the requested id is still
  honoured, so every link lands on one id and `wandb sync` merges them server-side.
  Continuity is carried by the id plus a monotone `global_step`, not by `resume=allow`.
- `WANDB_DIR="$RUN_DIR"`, not `$RUN_DIR/wandb` — wandb appends `wandb/` itself.
- Syncing **while a link is still running** ends with
  `ERROR Internal error: transactionlog: error reading: unexpected EOF`. Cosmetic: it is
  the live tail of a `.wandb` file still being appended. Measured — a first sync uploaded
  `history steps 0-23`, a second `24-53`. It resumes at the right offset, does not restart
  or duplicate, and converges once the link exits. Safe to run repeatedly mid-chain.
- `torch.load(map_location="cuda")` drags saved RNG ByteTensors onto the GPU and
  `set_rng_state` requires CPU. GPU resume breaks while CPU tests stay green.

### Reading the metrics — three traps that each cost a run

- **`Δ_op` reads as a dead operator when the *coefficients* die.** It compares a true
  operator against a negative, so a collapsed `c` makes the comparison vacuous. R0-A2's
  `delta_op → 0.0002` looked like a bank collapse; the bank had not moved at all
  (`r_mean` 0.7902 → 0.7899, `gnorm/bank` 0.01 at every step including a 71535 spike).
  Check `gnorm/bank` and `bank/live_ops` before blaming the bank.
- **Aggregate operator statistics hide the structure. Always split by replan *index*.**
  Run-wide entropy said "the policy picks one operator". Per-replan-index entropy said
  why: 2.47 at replan 0 (varying by task) then ~4.74 — near the `ln 128 = 4.852` ceiling —
  for every replan after. `z` stops depending on the observation after one recurrent
  update, and the policy runs one open-loop primitive per episode. `--op-stats` emits this.
- **Know each loss's degenerate floor before reading it as progress.** Uniform
  Plackett–Luce is `Σ log(128−i) = 19.361`; `act/align`'s disjoint-support MSE floor is
  `8·0.25² = 0.500`. A head pinned at exactly its floor looks like a plateau, not a
  failure. R0-A sat on both for 7004 steps.

### Gradient clipping does not protect an AdamW run

`clip_grad` bounds the norm, but **AdamW's update is invariant to a global rescale of the
gradient** — scaling `g` by `s` scales `m` by `s` and `√v` by `s`, so `m/√v` is unchanged.
A norm-71535 gradient clipped to 1.0 still takes a full LR step in the spike's direction,
holds it in momentum for ~1/(1−β₁) = 10 steps, and shrinks every *other* module's update by
that same 1e-4. Only skipping the step helps — `optim.spike_mult` (SpikeGuard), off by
default. Its threshold must be a **geometric** mean (the per-window median moves 1.5→29
over a run) and must update on skipped steps too via `min(gnorm, thresh)`: updating only on
accepted steps deadlocks, and one arm silently skipped 457 consecutive steps with nothing
in the loss curve to show it.
- Rank plumbing is `RANK=$SLURM_PROCID`, not torchrun. Without it every task reads rank 0
  and eight processes train the same shard while the loss curve looks fine.
- `scontrol` blocks indefinitely if slurmctld is unreachable from the node. Only call it
  when `SLURM_NNODES > 1`, wrapped in `timeout 30`.
- **`torch.compile` needs `export TRITON_CACHE_DIR="$PWD/.triton_cache"`.** Login nodes run
  glibc 2.35, compute nodes run **2.31**. The shared `~/.triton/cache` holds a
  `cuda_utils.so` linked against glibc ≥2.34, so every compile on a compute node dies with
  `GLIBC_2.34 not found` and reads like "inductor is unsupported here". It is not — a fresh
  project-local cache rebuilds against the node's own glibc and compiles fine (verified on
  an A100). **Never clear `~/.triton/cache`**; another project shares it.
- `/dev/shm` is **64 MiB on login nodes but 1008 GiB on compute nodes** — both measured. A
  DataLoader prefetch queue does not fit on a login node and fails as an opaque worker bus
  error, so local debugging misleads. Never hardcode `num_workers` from what you see on the
  login node; `loom.data.loader.fit_workers()` measures at runtime and is the right call.
- Lustre punishes small strided reads: fancy-indexed `np.memmap` measured 13 MiB/s where a
  coalesced `os.preadv` of the same bytes got 650 MiB/s.
- `import torch` off Lustre costs ~44 s wall with ~1 s of CPU. Run test suites as one
  pytest process, not one per test.
- torch defaults to 32 intra-op threads on these 64-core boxes and thrashes under
  concurrent load. `OMP_NUM_THREADS=8` took one suite from 249 s to 8 s.

## Do not build

Analysis notebooks, plots, ablation grids, diagnostic studies, pixel decoding, VAEs, video
DiTs, MCTS, config-validation layers, CLI progress bars. The deliverable is three tables.
