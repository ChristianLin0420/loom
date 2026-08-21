# LOOM experiment diary

Schema: `loom-experiment-diary-v1`

This is an append-only human decision ledger. Immutable machine receipts under
each run's control directory remain authoritative for config, source, data,
checkpoint, scheduler, evaluation, and result identities. Diary entries explain
why a round existed and how its result was interpreted; they never authorize a
checkpoint or mutate a running job.

## Entry schema

Each round has a stable `round_id` and chronological events. Allowed event types
are `PLAN`, `SUBMITTED`, `TRAINING_TERMINAL`, `EVALUATION_TERMINAL`,
`INTERPRETATION`, and `AMENDMENT`.

Every event records:

- `utc`: ISO-8601 UTC time, or `retrospective` for reconstructed history.
- `status`: planned/running/completed/failed/cancelled as appropriate.
- `why`: the prospective reason or retrospective interpretation.
- `method_delta`: what differs from the immediately preceding round.
- `fixed_protocol`: seed, endpoint/schedule, evaluation work, and selection rule.
- `authority`: plan/receipt/result hashes and job or W&B IDs when they exist.
- `result`: exact counts/rates and integrity state when known.
- `next`: the predeclared or newly reasoned next action.

Corrections are new `AMENDMENT` events. Existing event text is not silently
rewritten after a run is submitted.

---

## Round `r0a-dualcode-formal-s0-20260820-v2`

### PLAN — retrospective

- utc: retrospective (executed 2026-08-20 to 2026-08-21)
- status: completed training; formal decision ABORT
- why: test a fresh dual-code R0-A method with a direct convergence/health gate
  before formal evaluation.
- method_delta: fresh LOOM modules, frozen SigLIP cache, dual q_action/proposal
  decoder objective; formal schedule horizon 32,000 with a possible extension.
- fixed_protocol: seed 0, world 16; formal evaluation would have used seeds
  0/1/2 and 1,200 LIBERO episodes only after a gate PASS.
- authority: formal plan SHA-256
  `798a536cb466ecc275cc7a21da9bf09e30a16f92c1d6e1ea79afe1c3ae75cdaf`;
  training jobs `32586472`, `32586473`, `32586474`.

### TRAINING_TERMINAL — retrospective

- utc: 2026-08-21T00:33:54Z
- status: completed exact update 32,000; controller exit was terminal ABORT
- why: the direct receipt classified `health_gate_failed`. Training itself was
  numerically/execution clean; this was a scientific policy stop, not an
  infrastructure failure.
- result: 32,000 contiguous metrics rows and 16 endpoint shards. Failed health
  observations included zero `delta_op`, zero higher-horizon selection deltas,
  q_delta live-operator count 10, and nonconverged primary relative ranges.
- authority: direct receipt SHA-256
  `e7ca6cb243adfdf58e12c18fef35323a87815e266ff6bed1b449e009682ecb87`.
- next: formal descendants remained cancelled; a later user-authorized
  diagnostic evaluation could measure the fixed endpoint without reversing the
  ABORT.

### EVALUATION_TERMINAL — retrospective diagnostic

- utc: retrospective (completed 2026-08-21)
- status: completed, diagnostic only; formal ABORT unchanged
- why: measure the exact already-trained step-32,000 endpoint after explicit
  user authorization. An initial diagnostic launcher completed the episodes but
  rejected its own two-key/three-key `policy_kw` provenance expectation; a
  versioned adoption run authenticated the complete results without rerunning
  any episode.
- authority: adoption plan SHA-256
  `1563c8fc0b6be88fe27aff420ef5debe48223079c3a53a87c027f2b4ab21e802`;
  merged result SHA-256
  `d2763083b457f4f63c1c9a9552f924e0183e0b9535c62d972501808027270ec6`;
  merged receipt SHA-256
  `93bae9eb18355b42e8b5bed728ecfaab40fded0c6b21276443d2906896868aa7`;
  W&B summary run `4ed05cebda244bd8` in `loom-r0-e2e-scratch`.
- result: seed successes 178/180/192; exact 550/1,200, zero errors,
  45.8333% success. Historical paired delta +8.5833 percentage points with
  task-bootstrap 95% CI [1.75, 15.50]. Suite rates: spatial 54.0%, object
  45.0%, goal 71.3333%, long-horizon 13.0%.

### INTERPRETATION — retrospective

- utc: retrospective
- status: diagnostic threshold bundle FAIL; no promotion authority
- why: aggregate, paired-CI, seed-0, spatial/object/goal signals improved, but
  long-horizon success collapsed below the descriptive 24% floor. The dual-code
  route can execute useful behavior, while the operator representation and
  recurrent credit path remain the main repair target.
- next: make the operator repair prospective in a fresh fixed-endpoint lineage
  and remove convergence gating as a control variable. Always evaluate the
  predeclared endpoint if execution/checkpoint integrity holds.

---

## Round `r0a-operator-repair-fresh-s0-20260821-v1`

### PLAN — 2026-08-21T05:40:12Z

- status: planned; no jobs submitted
- why: test the operator/recurrent repair directly. The preceding dual-code
  endpoint improved aggregate success but failed representation health and
  long-horizon behavior. This round fixes the endpoint at update 32,000 and
  guarantees evaluation after integrity, so a health threshold cannot suppress
  the end-to-end measurement.
- method_delta: canonical `configs/r0a_operator_repair.yaml`; retain the dual
  q_action/proposal deployed decoder path; add dense q_delta semantic anchoring,
  effect/contrastive dynamics repair, recurrent-prefix training, weighted
  suite/task sampling, and per-head balance. Fresh LOOM modules only; frozen
  SigLIP cached features; no checkpoint or optimizer reuse.
- fixed_protocol: seed 0; exactly 32,000 optimizer updates on world size 16;
  six resumable 4-hour train links; checkpoint every 500; selection is only the
  predeclared update 32,000. Then one verified consolidation, three parallel
  singleton seeds 0/1/2, 400 exact LIBERO episodes each, and an exact 1,200
  merge. Evaluation outcome and all health values are descriptive and cannot
  block publication.
- W&B: project `loom-r0-operator-repair`; group and immutable run IDs will be
  frozen in the machine plan at submission.
- preflight: real cache contains 56,189 windows across 2,000 trajectories. The
  exact 32k suite schedule assigns 12,800 draws to long/`libero_10` and 6,400
  each to spatial, object, and goal. Recurrent prefix choices are near-balanced
  (7,938–8,056 draws); an actual burn-in-12 batch loaded successfully. An
  exhaustive real-cache 32k × 16-rank sampler audit found all 128/128 batch
  examples unique at every update. Spike rejection is disabled, so the endpoint
  means exactly 32,000 optimizer updates rather than 32,000 attempted updates.
- data/eval integrity preflight: the adapter-normalized exact 40-file/2,000-demo
  raw action receipt is `7c02425a…6fb2`; the manifest plus all 2,000 referenced
  feature binaries (76,017,192,576 bytes) hashes to `ba754444…d188`. Every train
  link reauthenticates both before and after execution. The immutable plan embeds
  the exact historical 1,200-row pairing snapshot (`c2a208e0…e9029`), so later
  training/evaluation never depends on mutable baseline files. LIBERO git HEAD,
  clean worktree, pinned Python package closure, BDDL/init-state trees, and all
  seven SigLIP eval snapshot targets are reauthenticated before and after each
  singleton evaluation. Seeds have disjoint runtime/TRITON directories.
- durability: `metrics.jsonl` is fsynced at checkpoint boundaries and an
  authenticated crash tail may only roll back to `LATEST`. The W&B health ledger
  persists across links/requeues; five consecutive failed 20-update log calls
  are an execution failure, while a successful call resets that counter. Either
  that failure publishes a durable terminal marker. After each train invocation
  returns, it publishes a post-asset-verification PENDING receipt before the
  asset scan begins, and only an exact post-hash COMPLETE receipt closes it; a
  failed or interrupted check therefore remains terminal even if external
  assets are later restored, while in-training preemption remains resumable. Each
  physical eval attempt binds immutable pre-attempt environment and checkpoint
  identities, reauthenticates both afterward, and content-address quarantines
  its episode rows before failing if either identity changes.
- reporting: the exact candidate 1,200-episode SR and per-suite/per-seed counts
  are always the output. The frozen paired historical delta/CI is context only;
  this round defines no success-rate threshold, pass/fail bundle, checkpoint
  selector, or promotion decision.
- runtime estimate: the prior lighter burn-in-0 fresh 32k run took 9h05m. This
  recipe averages prefix 6, adding no-grad estimator work and effect/contrastive
  bank work; six links provide about 23h usable headroom. Unused tail links
  authenticate step 32,000 and exit as zero-update no-ops. Retain 4h caps on
  every link/eval for queue and environment variance.
- authority: canonical config raw SHA-256 `7d4586f4…8143`, resolved hash
  `b47825f0cfba68dd`; source closure and plan/W&B IDs remain pending final freeze.
  This diary entry is not launch authority.
- next: focused orchestration regression, shell syntax checks, authenticated
  no-write dry-run, independent review, then commit/push and submit under the
  user's existing explicit authorization.
