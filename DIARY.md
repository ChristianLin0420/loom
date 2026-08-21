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

### SUBMITTED — 2026-08-21T07:44:15Z

- status: running; `train_01` began on two 8-GPU nodes and every descendant is
  held behind an exact `afterok` dependency.
- why: the frozen operator-repair implementation passed independent exact-byte
  review, 49 focused tests, 239 relevant checks (two expected skips), and a
  full authenticated no-write preflight. The user explicitly requested direct
  formal training followed by evaluation without a scientific gate or smoke
  run.
- method_delta: no change from this round's prospective PLAN. The submitted
  bytes are commit `5138a52` on branch `r0a-rerun-three-fixes`, pushed to
  `origin/r0a-rerun-three-fixes` before submission.
- fixed_protocol: six sequential resumable training jobs, one consolidation,
  three parallel exact-400 singleton evaluations, and one exact-1,200 merge.
  There are no decision-gate jobs; update 32,000 is the only checkpoint eligible
  for evaluation.
- authority: immutable plan SHA-256
  `a409ce347fe1c56fdf4bf01558d03111f3fb22a2e0c98b9219b923dfd452be8a`;
  jobs receipt SHA-256
  `d0ec335f9763446015533ac04b4ef725fd30d93a43cbe697b0595f97dade4ddc`;
  train jobs `32651388`–`32651393`; consolidate `32651394`; eval seeds 0/1/2
  `32651395`/`32651396`/`32651397`; merge `32651398`.
- result: submission released successfully. At this event, `32651388` was
  RUNNING and all ten descendants were dependency-pending; no training result
  or success rate existed yet.
- next: monitor every link through terminal execution, record material training
  and evaluation transitions as new diary events, and publish the exact
  end-to-end success rate without applying an outcome threshold.

### TRAINING_TERMINAL — 2026-08-21T19:10:20Z

- status: completed exact update 32,000; consolidation running
- why: execute the prospectively fixed operator-repair endpoint without metric
  selection. Four links performed updates; links five and six authenticated the
  completed endpoint as zero-update no-ops. No convergence or health value
  controlled stopping or evaluation eligibility.
- method_delta: none. The submitted config/source/data receipts remained exact
  across every link, and the optimizer/scheduler/sampler/W&B lineage resumed
  continuously at steps 10,663, 21,186, and 31,790.
- fixed_protocol: 32,000 optimizer updates, 32,000 contiguous metrics rows,
  world size 16, and the single predeclared step-32,000 checkpoint. All six
  training jobs completed `0:0`; no update was skipped and no numeric value was
  nonfinite.
- authority: fixed-endpoint receipt SHA-256
  `435838dec946835adbec33585dc75eef3c14607234a5f1c725af9df18e9ef597`;
  metrics SHA-256
  `3ceed51a2fe68e204e83363727f156e50922c86a719807ef9b20791dc1e46b79`;
  16 authenticated endpoint shards; W&B run `eb4b90e6d4c14e22` in project
  `loom-r0-operator-repair`.
- result: execution integrity PASS with zero skipped/nonfinite updates and zero
  W&B logging failures. Last-500 medians were q_delta/q_action live operators
  45/12, contrastive loss 3.83181, effect gap 0.39930, delta-op 0.01684,
  aggregate selection delta 0.02779, teacher/deployed decode about 0.04792,
  proposal CE 0.64073, and alignment/proposal top-k overlap 0.39844/0.43750.
  Training therefore formed real effect/selectivity signals and a broader
  transition codebook, but retained a narrower action codebook and imperfect
  semantic bridge. Large finite pre-clip estimator gradients remained a method
  caveat; clipping kept every update finite and applied.
- next: verify consolidation, evaluate seeds 0/1/2 for exactly 400 episodes each
  regardless of the training observations above, merge exactly 1,200 episodes,
  and report the raw end-to-end success rate plus descriptive per-suite/seed
  breakdowns without a threshold.

### AMENDMENT — 2026-08-21T19:12:00Z

- status: evaluation orchestration interrupted; training endpoint unchanged
- why: consolidation job `32651394` reconstructed and verified the endpoint
  checkpoint successfully, then the wrapper rejected its own receipt because it
  expected zero-padded shard suffixes (`rank00000`) while the authoritative
  checkpoint receipt uses real filenames (`rank0`). This is a validator-format
  defect, not a checkpoint/model failure.
- method_delta: none; zero optimizer or environment steps occurred after the
  fixed training endpoint.
- fixed_protocol: the canceled seed jobs and merge produced no episodes and no
  success-rate result. The recovery must remain exactly checkpoint verification
  followed by three 400-episode seeds and a 1,200-episode merge, with no smoke,
  training continuation, checkpoint change, or metric threshold.
- authority: failed consolidation job `32651394` exit `2:0`; its verification
  report passed all 923 keys, 3,920 replicated tensors, 10,808 sharded pieces,
  and real-policy loading with zero mismatches. The existing consolidated
  checkpoint is 1,760,597,436 bytes and will be rehashed and verified by a
  separately versioned recovery plan. Original eval jobs `32651395`–`32651397`
  and merge `32651398` were dependency-canceled before starting.
- result: no end-to-end SR yet. The source validator is corrected to use the
  real unpadded shard names and gains an executable regression for all 16 ranks.
- next: freeze and independently review an isolated zero-training recovery DAG,
  authenticate both the historical training closure and corrected runtime
  closure, verify/adopt the existing checkpoint, then evaluate unconditionally.

### SUBMITTED — 2026-08-21T20:17:58Z (evaluation recovery v2)

- status: running; checkpoint-adoption job started and four descendants are
  dependency-held
- why: resume only the evaluation work canceled by the rank-suffix validator
  defect. Independent review confirmed the recovery cannot call training,
  reconstruct a checkpoint, choose a checkpoint, or apply a scientific gate.
- method_delta: none. Commit `d153199` changes the validator from padded to the
  real unpadded shard suffix and adds a versioned evaluation-only wrapper. The
  source step-32,000 checkpoint remains byte-identical.
- fixed_protocol: one verify-only checkpoint adoption, three parallel exact-400
  seed evaluations, then one exact-1,200 merge; zero training jobs, zero
  consolidation/reconstruction jobs, and zero decision-gate jobs.
- authority: recovery plan SHA-256
  `c29f386efddf1bbd1987934eb075e7414b86c608badc7d55cd1999bbd618448c`;
  current 57-file runtime closure
  `614aafaac5fb48004a11ce5879ff590354921a91c1ff43b688cdbf5b2076eaf3`;
  source checkpoint SHA-256
  `ee8d3d583624be8c87cf6222c2d1716905d0ea21a4e1af5db094ef5d8273b36c`.
  Jobs: adopt `32687797`, eval seeds 0/1/2
  `32687798`/`32687799`/`32687800`, merge `32687801`.
- result: submission released successfully; no episodes had started at this
  event.
- next: monitor checkpoint adoption, every singleton evaluation, and the merge;
  report the exact raw end-to-end success rate with no outcome threshold.

### AMENDMENT — 2026-08-21T20:21:00Z

- status: recovery v2 stopped before evaluation; no episodes ran
- why: verify-only checkpoint adoption again passed, but the host-runtime
  receipt correctly noticed that the editable local package's `pip freeze` line
  changed from commit `d153199` to diary-only commit `93dc39c` after the plan
  was created. Executable source closure and checkpoint bytes did not change.
- method_delta: none; no training, reconstruction, selection, or environment
  interaction occurred.
- fixed_protocol: jobs `32687798`–`32687801` were dependency-canceled before
  start. A fresh recovery plan will use the current Git HEAD, while diary edits
  remain uncommitted until all evaluation jobs and merge are terminal.
- authority: adoption job `32687797` failed `2:0` only after a fresh checkpoint
  verification `OVERALL PASS`; source closure remained
  `614aafaac5fb48004a11ce5879ff590354921a91c1ff43b688cdbf5b2076eaf3`.
- result: no success rate exists yet.
- next: resubmit the same five-stage zero-training recovery from stable HEAD
  `93dc39c` with fresh roots, then defer every Git commit until merge completes.

### SUBMITTED — 2026-08-21T20:23:34Z (evaluation recovery v3 roots)

- status: running; no Git commits are permitted until terminal merge
- why: repeat the already-reviewed recovery from stable editable-package HEAD
  `93dc39c`, eliminating the v2 host-receipt drift while preserving identical
  executable bytes and checkpoint.
- method_delta: none; this changes only fresh output roots, plan/run IDs, and
  the host-runtime receipt derived from the stable current HEAD.
- fixed_protocol: verify-only adoption `32688197`, parallel exact-400 evals
  `32688198`/`32688199`/`32688200`, exact-1,200 merge `32688201`; no training,
  reconstruction, checkpoint selection, or outcome gate.
- authority: recovery plan SHA-256
  `9d3590cc96ccef239f61e529b7bd82f4007895800cb222f9155fd95ce0280717`;
  jobs receipt SHA-256
  `c02535080ba3ef6d2670ae8a75b2df4790fab4d104358dcaa4659ff5a6c9d6aa`.
- result: adoption was RUNNING and all evaluation/merge descendants were held at
  this event; zero episodes existed.
- next: make no repository commit, monitor all five jobs, then append and push
  the final evaluation/interpretation only after merge is immutable.

### EVALUATION_TERMINAL — 2026-08-21T20:57:36Z

- status: completed; all five recovery-v3 jobs exited `0:0`, and the exact raw
  end-to-end success rate is **66/1,200 = 5.50%** with zero episode errors.
- why: finish the evaluation canceled by the consolidation receipt-format bug,
  using the already-fixed step-32,000 checkpoint without any training,
  reconstruction, checkpoint selection, smoke run, scientific gate, or outcome
  threshold. The verify-only adoption, three singleton evaluations, and merge
  all reauthenticated their inputs before publishing results.
- method_delta: none. The evaluated model is byte-identical to the endpoint
  produced by this round's 32,000-update operator-repair training. Recovery v3
  changed only isolated receipts/output roots and performed zero optimizer
  steps.
- fixed_protocol: checkpoint adoption job `32688197`; exact-400 seed jobs
  `32688198`/`32688199`/`32688200`; exact-1,200 merge `32688201`. Seed results
  were 20/400 = 5.00%, 26/400 = 6.50%, and 20/400 = 5.00%, respectively, all
  with zero errors.
- authority: recovery plan SHA-256
  `9d3590cc96ccef239f61e529b7bd82f4007895800cb222f9155fd95ce0280717`;
  source checkpoint SHA-256
  `ee8d3d583624be8c87cf6222c2d1716905d0ea21a4e1af5db094ef5d8273b36c`;
  completion SHA-256
  `829442e84df6733d4b77c7227d25b0757b75042b117370408bf744d7c87d3760`;
  merged-receipt SHA-256
  `14800f951c4af858cfd25390b4a9ec59893f21ebd56f6567d64e0daa8bdb46d9`;
  result SHA-256
  `9e6e462e463d9e0d042a9fa40dda385cf440b3d6eced969e24f8065be47fb17c`;
  table SHA-256
  `2604dc3bf524ee4397999ac80cd5493fc73c1c98883013e4963c2d78a5ca94bf`.
- result: per-suite SR was Spatial 5/300 = 1.67%, Object 43/300 =
  14.33%, Goal 8/300 = 2.67%, and Long 10/300 = 3.33%. A descriptive paired
  comparison to the frozen 447/1,200 = 37.25% baseline gives -31.75 percentage
  points with a 95% paired task-bootstrap interval of [-37.00, -26.25]
  percentage points. Exactly 1,134/1,200 episodes reached the 512-step cap.
  These statistics did not control execution or publication.
- integrity: completion records `training_updates=0`, `optimizer_steps=0`,
  `checkpoint_reconstructions=0`, `scientific_gates=0`, and
  `outcome_threshold_applied=false`. The online summary is W&B run
  `53259d852bef4d32` in project `loom-r0-operator-repair`.
- next: explain why the operator-training gains failed to translate into the
  deployed policy and retain the evidence as a negative method result rather
  than extending or relabeling this run.

### RESULT_INTERPRETATION — 2026-08-21T21:00:00Z

- status: method failure for end-to-end control. The 5.50% SR is not an
  infrastructure artifact or an ambiguous statistical result; it is a large,
  consistent regression across all three seeds and all four suites. Relative
  to the previous dual-code checkpoint's 550/1,200 = 45.83%, this round lost
  40.33 percentage points; 19/40 tasks had zero successes.
- why: the repair solved part of the original LOOM representation problem but
  coupled it to the deployed action teacher in a way that destroyed action
  coverage. The fixed evaluation path executes estimator -> proposal ->
  decoder; it does not call the learned bank or `q_delta`. Consequently, better
  transition-effect and selection metrics could not directly improve actions,
  while damage to `q_action` and the proposal/decoder path directly reduced SR.
- training_evidence: in the terminal 2,000 updates, effect gap was 0.3995 and
  positive on 2,000/2,000 rows; aggregate selection delta was 0.02768 and
  positive on 1,996/2,000 rows; all four horizons were simultaneously positive
  on 1,903/2,000 rows. These are real dynamics-discrimination signals rather
  than the earlier zero-margin collapse, but evaluation shows they encode a
  compressed shortcut rather than a broad semantic operator vocabulary.
  Recurrent prefixes also improved effect, selection, alignment, and proposal
  metrics monotonically, so the added history path was useful in isolation.
- failure_evidence: the apparently 45-live-operator `q_delta` distribution had
  only 5.13 effective atoms by entropy, and `q_action` had only 4.11, versus
  about 50.6 effective `q_action` atoms in the prior 45.83%-SR run. The new
  teacher/deployed decode losses were 0.0481/0.0483 versus 0.0167/0.0228
  previously; proposal top-4 overlap fell from 0.711 to 0.438. The lower raw
  proposal CE (0.639) mostly tracked a much lower-entropy teacher rather than a
  richer action policy. Evaluation independently observed only 11/128 top-1
  codes and 3.775 effective codes; the top five carried 98.31% of selections,
  and task/code mutual information was only 5.74% of code entropy. Four
  back-and-forth transitions among dominant codes accounted for 67.52% of all
  operator switches. This repetitive, task-agnostic behavior is consistent
  with 94.5% of episodes timing out at the step cap.
- likely_mechanism: dynamics was configured with `coeff_source=q_action` and
  `detach_coeff=false`, so state/effect/contrastive gradients reshaped the same
  action-code teacher used by proposal and decoder. Terminal pre-clip gradient
  norm was dominated by the estimator (median 155.11 of 155.24 total, about
  99.88% by norm ratio); every terminal-window update was globally clipped,
  while `q_delta` received a median norm of only 0.0796. The most likely failure
  is therefore cross-objective interference plus global-clipping starvation,
  followed by action-code compression. This is a causal hypothesis supported
  by the chronology and routing, not a claim that scalar logs recover gradient
  directions exactly.
- next_method: preserve the original LOOM idea but separate representation
  formation from action realization. First form a broad `q_action` + decoder +
  proposal policy using action supervision. Protect that action anchor from
  dynamics gradients; drive dynamics from `q_delta`, and train `q_delta` + bank
  against dense `q_delta -> stopgrad(q_action)` semantic alignment plus
  effect/contrastive losses. Use separate optimizer groups, gradient projection,
  or per-loss/module clipping so estimator dynamics cannot monopolize the
  update. Only after both routes are stable should a low-LR joint phase align
  them, with explicit marginal-entropy/task-information preservation to prevent
  code collapse. Finally, the evaluated LOOM policy must actually use bank
  rollouts or outcome/potential reranking; otherwise operator learning remains
  a training auxiliary and cannot provide the project's intended planning
  benefit.
- result: do not continue this same objective for more updates and do not treat
  its strong operator diagnostics as a proxy for SR. Archive 5.50% as the exact
  end-to-end result and use the decoupled staged recipe for the next prospective
  round.
