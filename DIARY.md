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

## Round: protected-action parallel arms

### PLAN — 2026-08-22T03:13:53Z

- status: prospective design; implementation and immutable receipts pending
- why: the prior dual-code run achieved 550/1,200 = 45.83% because its deployed
  `q_action -> proposal -> decoder` route retained about 50.6 effective action
  atoms. The operator-repair run let dynamics gradients rewrite that route,
  reduced it to about four effective atoms, and scored 66/1,200 = 5.50% even
  though bank-effect diagnostics improved. The next comparison must protect the
  executable action vocabulary while testing whether LOOM dynamics can learn in
  a separate route.
- shared_protocol: three fresh seed-0 lineages from identical random
  initialization and identical step-indexed samples; real modules; frozen
  cached SigLIP tower; dual teacher/deployed decoder; 20/20/20/40
  Spatial/Object/Goal/Long sampling; recurrent-prefix choices 0/4/8/12; exact
  update 32,000; online W&B project `loom-r0-protected-arms`; fixed checkpoint
  only; exact seeds 0/1/2 x 400 episodes; raw exact-1,200 merge. Training
  diagnostics cannot stop, extend, select, or suppress evaluation. There is no
  SR threshold, convergence gate, smoke-training run, or promotion decision.
- arm_H_history_anchor: preserve the successful dual-code loss semantics:
  action-free `q_delta` state/hinge dynamics, MSE
  `q_delta <- stopgrad(q_action)`, pooled Switch balance, sparse proposal CE,
  and paired teacher/deployed CFM. Its only method package relative to the
  45.83% external anchor is the shared Long-aware sampler and real recurrent
  prefixes. This determines whether those data changes preserve the strong
  direct policy before attributing differences to operator repair.
- arm_P_protected_operator: keep H's data and direct action path, but replace
  the operator package with dense sparse-target CE
  `q_delta <- stopgrad(q_action)`, per-head balance, and residual-effect plus
  contrastive dynamics. Dynamics consumes attached `q_delta`, never
  `q_action`; therefore it can train the transition vocabulary but cannot
  directly overwrite the executable action teacher. It uses the fixed
  updates-1..2,000 formation phase and updates-2,001..2,500 dynamics ramp.
- arm_I_isolated_operator: exactly P plus one gradient-routing change:
  operator-side beliefs are detached before `q_delta`/bank/dynamics, including
  alignment and balance. Estimator gradients remain live through the direct
  `q_action`/proposal/dual-decoder route. This tests whether the P arm's
  remaining shared-estimator edge recreates the prior 99.88%-dominated global
  clipping failure.
- comparison_logic: H versus the frozen 45.83% checkpoint measures the shared
  data/history package; P-H measures protected operator learning; I-P measures
  estimator isolation. All three results will be published, not merely the
  best arm. Warm-starting from the 45.83% checkpoint is excluded because it
  would change optimizer/LR/global-step history and obscure these causal
  comparisons.
- inference_scope: the exact scored endpoint remains the deployed R0
  estimator -> proposal -> decoder policy for comparability. Existing code has
  no trained goal-conditioned potential for bank rollout selection, so this
  round will not fabricate an arbitrary bank-reranking score. A genuine
  bank-assisted endpoint requires a separately frozen scorer method after the
  protected representation is established.
- authority: exact config/source/plan hashes, run paths, W&B IDs, and Slurm job
  IDs will be appended only after tests, authenticated no-write dry-runs,
  independent review, commit, and push. This PLAN is not launch authority.
- next: implement the three configurations and the single versioned parallel
  orchestrator, verify gradient routing and exact resume/evaluation behavior,
  then freeze, commit, push, and submit all arms without consulting training
  metrics.

### IMPLEMENTED_AND_REVIEWED — 2026-08-22T03:48:52Z

- status: launch candidate frozen; no training or evaluation job has yet been
  submitted. The implementation uses one authenticated profile wrapper and
  the previously exercised fixed-endpoint resume/consolidate/evaluation
  machinery, rather than three divergent orchestration copies.
- method: H retains the prior dual-code loss and gradient semantics. P routes
  state/effect/contrastive dynamics through attached `q_delta` and uses
  `q_action` only as a stop-gradient semantic target. I adds estimator
  isolation for every operator-side path. A pre-launch cross-review found that
  I's per-head `q_action` balance term still reached the estimator; the final
  implementation recomputes that small balance forward from a detached belief,
  so balance continues to train `q_action` parameters while contributing
  exactly zero estimator gradient. The live action decoder/proposal forwards
  remain attached and continue training the estimator.
- exact_configs: H raw SHA-256
  `39befbbdc2bfacf62aef4dd9c890bf2747ff3142fcc63ea9769641328defdcce`,
  resolved hash `7c79c09af56ccf16`; P raw
  `145e41e0201dd80dc95b0beb4325f26ae3795a39ded0270ff1cb9aa2a1e45948`,
  resolved `08ca78d71dc0321b`; I raw
  `71ac5d48b6b7b2f6f15ab098c3ec2063402c544bec4ae12deaad023985a0a73a`,
  resolved `3cb3dea37bcb9aaf`. The common config SHA-256 is
  `a8b386bb4645e1e5f14bb357d96c6b5932bb791686129bc76432a4d74cff1f3e`.
- exact_code: protected wrapper SHA-256
  `644a8b1fcad82b0133423156c75772a954a87ea13aa4b76a02826d54a53da775`;
  training-loop SHA-256
  `ff4e8995e146ee4913dcd9cdd07c42dc5b5d58249efd94d3b3fe08bcbc57fa29`;
  protected profile-table SHA-256
  `a438d1c709108441532230238d2d092bfc9f8fd53c9f0d0a0eb7e055403047b7`;
  ordered 59-file execution-closure SHA-256
  `b7fe3f91452a4c39cb3bea9fd5e92eccef2461a9ad22a0f63f0800e0b24e70d3`.
- verification: independent method and orchestration reviews both returned
  PASS. The final focused protected/legacy suite passed 71/71; the broader
  protected/operator/full-train compatibility run exited zero; Python compile,
  all four protected Slurm shell syntax checks, and `git diff --check` passed.
  Full authenticated H/P/I no-submit dry-runs each exited zero and created no
  run/control/artifact directories. The evaluator package identity remains the
  pre-existing pinned 118-line receipt; repository source is authenticated by
  the separate 59-file closure.
- execution_contract: each arm has six sequential resumable 2-node/16-GPU
  training links, then checkpoint consolidation, three parallel 400-episode
  seeds, and an exact 1,200-row merge: 11 jobs per arm, 33 total. All stages
  use the new online W&B project `loom-r0-protected-arms` with exact arm-specific
  group/tags. There is no convergence, health, SR, checkpoint-selection, or
  cross-arm selection gate; finite integrity/execution failures alone may stop
  descendants.
- next: commit and push these exact design/code bytes, submit H/P/I to fresh
  isolated roots, record plan/job/W&B identities, and monitor all three through
  their unconditional evaluations.

### SUBMITTED — 2026-08-22T03:57:24Z

- code: commit `0123b90` (`Add protected-action parallel training arms`) was
  pushed to `origin/r0a-rerun-three-fixes` before any plan was materialized.
  The pre-existing generated `MUJOCO_LOG.TXT` modification was deliberately
  excluded from that commit. No code/source commit is permitted while these
  authenticated plans are active.
- arm_H: plan SHA-256
  `a98722c35628d80b80bba8b5e46959d387e1f44c7ab32f4045c2709bdbde29a6`;
  jobs `32710174` through `32710184` (train 01..06, consolidate, eval seeds
  0/1/2, merge); jobs-receipt SHA-256
  `0e72a11a1af142d3af2bd0fe0d12bab3644a67503a73786c167d8b2fda93bd56`;
  release SHA-256
  `68fab1410f4c779715da0f9c05b9b9a5abc55a7398b843b26b30c1cbd01f9bc1`;
  W&B training ID `ab2676cc60934731`, group
  `r0a-protected-fixed32k-s0-20260822-arm-h-v1`.
- arm_P: plan SHA-256
  `7110196b2169e971032cbdfd8cf5642fcd1d8a5705c2b1c6f274eec28ceb9fb7`;
  jobs `32710273` through `32710283`; jobs-receipt SHA-256
  `5a72bbd9ecbec48f9d9ff2db90d362ab04bdecdfeac314801649bf4f066324a5`;
  release SHA-256
  `2b84905e08198e8016ccd078cf7bd951bda5882851dd769df7c7c6100a50d498`;
  W&B training ID `149fb4c5bbba4a4d`, group
  `r0a-protected-fixed32k-s0-20260822-arm-p-v1`.
- arm_I: plan SHA-256
  `764f331fb3eda9ddece404f4c312ad21094c0ec272e7ede0398534e67dd878f6`;
  jobs `32710313` through `32710323`; jobs-receipt SHA-256
  `414d4aefaf1f9695f0548bf4cdd97bc62e1314932118158b28b84483627f20ad`;
  release SHA-256
  `18b165867d4d7639ad4749dd7617a6c1b7f01de0c7ba33a7d689e6a761216f45`;
  W&B training ID `d148c34b0b1741f7`, group
  `r0a-protected-fixed32k-s0-20260822-arm-i-v1`.
- launch_state: all three 11-job DAGs were released with empty
  `decision_gate_jobs`. H train-01 began on two 8-GPU nodes at
  2026-08-22T03:56:40Z; P and I train-01 were scheduler-pending for Priority at
  this observation. Their dependency chains are held behind exact `afterok`
  edges, not a scientific threshold.
- next: monitor all three lineages, record operational/method observations as
  descriptive evidence only, and allow every integrity-valid arm to reach the
  fixed 32k endpoint and exact 1,200-episode evaluation.

### EXECUTION_FAILURE_AND_FIX — 2026-08-22T04:00:31Z

- result: v1 produced no scientific training or evaluation result. H train-01
  job `32710174` failed after 82 seconds, before model initialization, because
  every distributed rank executing `scripts/r0_e2e_protected_train_entry.py`
  as a file raised `ModuleNotFoundError: No module named 'scripts'`. The entry
  worked in import-based tests because the repository root was already on
  `sys.path`; the real `python scripts/...` invocation sets `sys.path[0]` to
  the `scripts` directory. This was an orchestration import defect, not a model,
  data, optimizer, or W&B failure.
- containment: the exact 33 v1 job IDs were canceled once the shared failure
  mode was confirmed. H had one FAILED job and ten dependency-canceled jobs; P
  train-01 was allocated for 27 seconds and user-canceled before reaching an
  update, with ten dependency-canceled descendants; all eleven I jobs were
  canceled without starting. Across H/P/I there is no `metrics.jsonl`,
  `LATEST`, model checkpoint, evaluation result, or optimizer update. The v1
  roots and receipts remain untouched as the failure record and will not be
  reused.
- fix: the protected training entry now inserts its resolved repository root
  into `sys.path` before importing the shared strict-W&B entry. A new
  subprocess regression executes the entry exactly as Slurm does with
  `PYTHONPATH` absent; it proves import succeeds and reaches the expected
  protected-arm environment validation instead of `ModuleNotFoundError`.
  Entry SHA-256 is
  `953cefc87c260ebf0c91263f80a90cabd5b6830eb2dd46a0a28c9a6573d42dd4`;
  test SHA-256 is
  `82b1642be2b37b5c3b456094c5983d5f8c700d442a17a7eaafed387ebdaedd4f`;
  the updated 59-file closure is
  `690e018045a964790507307ee8125181324c0f3b6d6a717ca304aa6cb0a76836`.
- verification: the direct sanitized executable check reaches
  `LOOM_PROTECTED_ARM must be exactly one of H/P/I`, and the protected,
  legacy-operator, and method test slice passes 72/72. Independent narrow
  review, commit/push, full asset-authenticated v2 dry-runs, and fresh v2 roots
  are required before resubmission.
- why_relaunch: no update was taken, so there is no partial lineage or method
  result to salvage. Fresh v2 plans preserve the predeclared H/P/I comparison
  without contaminating optimizer or sample history.

### RESUBMITTED_V2 — 2026-08-22T04:13:22Z

- code: direct-entry fix commit `2e63f55` was pushed to
  `origin/r0a-rerun-three-fixes`. All H/P/I full asset-authenticated no-write
  v2 builds passed first and created none of the target roots. The three new
  plans share updated 59-file closure SHA-256
  `690e018045a964790507307ee8125181324c0f3b6d6a717ca304aa6cb0a76836`.
- arm_H_v2: plan SHA-256
  `1a17688b281ac1620e3a6cf8a950439d877c119e71e363ca39faba6ee165be48`;
  jobs `32710459` through `32710469`; jobs/release SHA-256
  `a90372f4b19193363e13246c81e56f79608343c7e2861a534fbcf8c5440be8e7` /
  `9faf05a3c46653324c0944aef0d8427fc26142135905d775811052baa5dbc40c`;
  W&B training ID `bb865fe0fa68442f`.
- arm_P_v2: plan SHA-256
  `70a83d576fff8af54e44e6be48ffa2323a0743f2d04fffdb2e6041db84d6a6fb`;
  jobs `32710607` through `32710617`; jobs/release SHA-256
  `7759347b91582ebab4221362f84aafd55f8b2fb1fd082e975eb47605bec836ee` /
  `9850f20acbe9fa31023c540b69b128a4f5a1dd52f42e696e1e7cb116443d5110`;
  W&B training ID `a1d8a5066557422d`.
- arm_I_v2: plan SHA-256
  `8bad6e0d1b76d4d78adcceca96b85d0dd0691df02389ef42bdd5afebfb268d13`;
  jobs `32710736` through `32710746`; jobs/release SHA-256
  `c1c732249c0d393151324e852a2296f60f0e59827929720022f87ddca33cb3d9` /
  `bb1985e3b602d8d252f48382d662205c516fb9f8a5b3a894c3218a5ffdaaa89a`;
  W&B training ID `7ad4e3ffe04842a4`.
- launch_state: H cleared the repaired direct-file import, built all real
  modules/FSDP/loaders on 16 ranks, connected online W&B, and had 111 contiguous
  finite update rows at this observation. P train-01 had allocated and was in
  initialization; I train-01 was Priority-pending. Each v2 DAG again has no
  decision-gate job and uses an exact fixed 32k endpoint plus unconditional
  3x400 evaluation.
- next: regard v2 as the only scientific lineage, monitor numerical and
  orchestration integrity, and record H/P/I method metrics descriptively
  without changing execution.

### RELAUNCH_HEALTHY — 2026-08-22T04:17:08Z

- status: all three train-01 jobs are RUNNING on two nodes/16 ranks and have
  crossed the failed v1 boundary into real optimizer updates. At this frozen
  observation H/P/I have 288/166/81 contiguous metric rows respectively, all
  with finite total loss and `grad_skipped=0`.
- online_logging: H/P/I training dashboards are respectively
  `https://wandb.ai/crlc112358/loom-r0-protected-arms/runs/bb865fe0fa68442f`,
  `https://wandb.ai/crlc112358/loom-r0-protected-arms/runs/a1d8a5066557422d`,
  and `https://wandb.ai/crlc112358/loom-r0-protected-arms/runs/7ad4e3ffe04842a4`.
- interpretation: this closes the v1 launch incident only; it is not a method
  or convergence claim. P/I dynamics and `dyn/estimator_isolated` are absent
  before their predeclared update-2001 activation, while H dynamics is active
  from the start. No observed value changes the fixed execution plan.

### OBSERVATION_000500 — 2026-08-22T04:26:09Z

- integrity: the comparison freezes exactly contiguous updates 1..500 for each
  arm. All three have zero skipped/nonfinite updates and exact online W&B
  acknowledgements every 20 updates through 500. It is descriptive only.
- H_medians: `q_delta` live/entropy 6/1.556; `q_action` 38/3.232; MSE align
  0.519; proposal/deployed top-4 overlap 0.117; proposal CE 4.558; decode
  0.278; estimator/bank/`q_delta`/`q_action`/decoder/proposal gradient norms
  1.116/0.00713/16.714/0.0300/1.452/4.652; global norm 18.05; loss 6.011.
- P_medians: `q_delta` live/entropy 6/1.554; `q_action` 55/3.723; sparse-CE
  align 4.834; proposal/deployed overlap 0.0625; proposal CE 4.790; decode
  0.276; gradients 0.0348/not-active/0.940/0.0117/1.472/2.381; global 3.313;
  loss 5.588.
- I_medians: `q_delta` live/entropy 6/1.583; `q_action` 43/3.473; sparse-CE
  align 4.844; proposal/deployed overlap 0.0391; proposal CE 4.837; decode
  0.274; gradients 0.000602/not-active/1.523/0.0195/1.471/1.690; global
  3.022; loss 5.664.
- why: P/I dynamics and bank updates are intentionally inactive through update
  2,000, and H uses MSE while P/I use sparse CE, so raw total/align values are
  not same-objective quality rankings. The intended routing distinction is
  already visible: I suppresses estimator leakage by about 58x versus P while
  retaining decoder/proposal/`q_action` gradients. All three still use a narrow
  median-six `q_delta` support this early. No metric changes execution; the
  informative next window follows the fixed 2,001..2,500 dynamics ramp.

### OBSERVATION_002000 — 2026-08-22T04:55:44Z

- integrity: every arm has exact `LATEST=2000`, a contiguous ledger, no skipped
  update, and healthy W&B. P/I activate dynamics at exactly update 2,001 with
  initial scale 0.002; I reports estimator isolation enabled and P disabled.
- H_boundary: `q_delta` live/entropy 6/1.565, `q_action` 42/3.357, MSE align
  0.659, top-4 overlap 0.219, proposal CE 3.516, decode 0.108; estimator/bank/
  `q_delta`/`q_action` gradients 0.0361/0.00855/11.963/0.00844.
- P_boundary: `q_delta` live/entropy 56/3.732, `q_action` 54/3.680, sparse-CE
  align 3.121, overlap 0.352, proposal CE 3.619, decode 0.098; gradients
  0.210/not-active/0.288/0.0139.
- I_boundary: `q_delta` live/entropy 34/3.176, `q_action` 62/3.706, sparse-CE
  align 3.977, overlap 0.133, proposal CE 4.078, decode 0.130; gradients
  0.0642/not-active/0.643/0.0199.
- interpretation: before any repaired dynamics update, P/I's dense semantic
  alignment preserves substantially broader transition-code support than H's
  legacy path (56/34 versus 6 live codes); P currently has the strongest
  alignment. This is a formation result, not end-to-end evidence. The decisive
  next comparison asks whether P retains that structure once dynamics is live
  and whether I avoids the earlier estimator-driven collapse.

### OBSERVATION_002500 — 2026-08-22T05:07:18Z

- integrity: exact ramp windows 2,001..2,100 and 2,251..2,500 are complete and
  contiguous for P/I; the predeclared dynamics scale moves from median 0.101 to
  0.751 and equals 1.0 at update 2,500. All arms remain zero-skip with healthy
  checkpoint and W&B ledgers.
- P_early_to_late: `q_delta` live/entropy 50/3.623 -> 49/3.630 and `q_action`
  57/3.615 -> 58/3.647; align CE 2.774 -> 4.336, overlap 0.453 -> 0.430,
  proposal CE 2.649 -> 2.718, decode 0.071 -> 0.064. State loss improves
  1.786 -> 0.979 and positive cosine 0.437 -> 0.699; effect 1.626 -> 1.495,
  while contrastive remains near-random 4.415 -> 4.410. `delta_op`/selection
  stay small positive near 2.4e-4/5.0e-4. Estimator gradient rises 0.149 ->
  1.360 while bank/`q_delta` rise 0.000422/0.404 -> 0.00288/3.497.
- I_early_to_late: `q_delta` 35/3.255 -> 42/3.440 and `q_action` 55/3.490 ->
  57/3.551; align CE 3.718 -> 4.326, overlap 0.297 -> 0.359, proposal CE
  3.279 -> 3.045, decode 0.085 -> 0.071. State/effect/cosine move
  1.996/2.645/0.144 -> 1.877/2.673/0.160; contrastive 4.424 -> 4.416;
  selection rises 8.1e-5 -> 2.55e-4. Estimator gradient remains isolated and
  flat 0.0416 -> 0.0373 while bank/`q_delta` grow 0.000548/0.628 ->
  0.00309/1.335.
- H_reference_2001_2500: legacy H has `q_delta` live/entropy 5/1.463 versus
  `q_action` 38/3.172, MSE align 0.679, overlap 0.375, proposal CE 2.829,
  decode 0.082, and near-zero selection. Its legacy state/cosine numbers are
  not loss-equivalent to P/I.
- interpretation: both protected arms avoid H's immediate transition-code
  collapse and keep the deployed route viable. P learns state/effect fitting
  faster but opens the shared-estimator gradient path strongly; I proves the
  intended isolation while learning operator effects more slowly. Neither has
  learned a meaningful contrastive margin yet. This is the designed P-I
  tradeoff, not a selection decision; all arms continue to fixed 32k.

### OBSERVATION_004000 — 2026-08-22T05:43:55Z

- integrity: exact contiguous stationary window 3,501..4,000, 500 rows per arm,
  zero skipped updates. Each arm has `LATEST=4000`, exactly four retained
  checkpoints x16 rank shards, healthy W&B, and ample filesystem headroom.
- H: `q_delta` live/entropy 5/1.534 versus `q_action` 45/3.402; MSE align
  0.629; deployed overlap 0.488; proposal CE 2.501; teacher/deployed decode
  0.0426/0.0694; coefficient L2 0.315. Legacy state/cosine 0.241/0.955 and
  near-zero `delta_op`; estimator/`q_delta`/proposal gradients
  0.0377/11.158/2.315, global 11.44.
- P: `q_delta` 28/2.716, `q_action` 61/3.727; align CE 4.789, overlap 0.438,
  proposal CE 2.742, teacher/deployed decode 0.0402/0.0684, coefficient L2
  0.344. Effect/state/cosine 1.016/0.720/0.735, contrastive 4.420, selection
  9.94e-5. Estimator/`q_delta`/proposal gradients 3.171/16.757/2.254,
  global 17.37.
- I: `q_delta` 28/2.855, `q_action` 60/3.732; align CE 4.729, overlap 0.438,
  proposal CE 2.780, teacher/deployed decode 0.0411/0.0681, coefficient L2
  0.349. Effect/state/cosine 2.501/2.059/0.110, contrastive 4.419, selection
  6.25e-4. Estimator/`q_delta`/proposal gradients 0.0320/5.277/2.265,
  global 5.86.
- interpretation: P and I now have essentially matched direct-policy metrics;
  both retain much broader `q_delta` support than H despite contracting from
  ramp end. P's shared estimator learns operator prediction faster but is
  driven roughly 100x harder than I; I maintains the intended estimator
  protection with weaker effect prediction. Contrastive separation remains
  near chance in both. All arms continue unchanged.

### OBSERVATION_008000 — 2026-08-22T07:03:02Z

- integrity: exact 7,501..8,000 windows, 500 contiguous rows per arm, zero
  skipped updates. Every arm has exact `LATEST=8000`, 64 retained shards,
  continuous metrics, and zero W&B failure events.
- H: `q_delta` remains collapsed at 5 live/entropy 1.547 while `q_action`
  reaches 49/3.52. MSE align 0.619, deployed overlap 0.539, proposal CE 2.366,
  teacher/deployed decode 0.0329/0.0575, coefficient L2 0.281. Legacy
  state/cosine 0.285/0.947. Estimator/bank/`q_delta`/proposal gradients
  0.0884/0.0460/10.016/1.919, global 10.24.
- P: `q_delta` contracts from 28 at 4k to 20 live/entropy 2.225 while
  `q_action` remains 62/3.719. Align CE worsens 4.789 -> 6.308, overlap
  0.438 -> 0.328, proposal CE 2.742 -> 3.168, coefficient L2 0.344 -> 0.462;
  teacher decode improves to 0.0349 but deployed worsens to 0.0749. State fit
  becomes excellent (0.050/cosine 0.981), yet effect is 1.936, effect gap only
  0.00248, contrastive 4.374, and selection near 4e-5. Estimator/proposal
  gradients rise to 7.29/6.14, with global 12.57.
- I: `q_delta` similarly contracts to 20/2.432 while `q_action` broadens to
  64/3.818. Align CE/overlap 4.830/0.461, proposal CE 2.704, teacher/deployed
  decode 0.0327/0.0581, coefficient L2 0.314: all materially healthier than P's
  deployed route at this point. Isolated operator prediction remains weak
  (state/effect/cosine 2.110/2.476/0.140, gap 0.00017, contrastive 4.423), but
  estimator gradient is 0.0503 versus P's 7.29; global gradient is 5.42.
- interpretation: P learned a very strong generic next-state fit, but shared
  estimator pressure now coincides with deterioration in the exact deployed
  proposal/action route. I demonstrates the intended protection and improves
  the direct route while sacrificing operator fit. H still has the best direct
  surrogate but an unusable transition vocabulary. These are mechanistic
  findings, not checkpoint selection; all three continue to fixed 32k.

### TRAIN_LINK_01_COMPLETE — 2026-08-22T08:03:48Z

- status: H/P/I train-01 jobs completed `0:0` after 03:46:01 / 03:45:33 /
  03:45:36 and committed exact steps 10,615 / 10,700 / 10,733 respectively.
  All three train-02 jobs are RUNNING on 16 ranks.
- resume_integrity: each link-02 uses the same plan/config/source lineage and
  same online W&B run with `resume=must`. Metrics append uniquely from the next
  update with no gap or duplicate; at this observation H/P/I ledgers extend to
  10,961 / 10,888 / 10,882. Pre/post raw/cache asset transactions and final
  checkpoint commits succeeded for all first links.
- interpretation: this is an operational durability milestone only. The
  different safe-budget endpoint steps are walltime effects, not checkpoint
  selection, and all lineages continue on the same fixed 32k schedule.

### OBSERVATION_P_GRADIENT_INSTABILITY — 2026-08-22T08:14:00Z

- integrity: H/P/I train-01 jobs remain `COMPLETED 0:0`; all train-02 jobs are
  `RUNNING` with contiguous, unique metric ledgers and the same online W&B IDs
  in `resume=must` mode. Every recorded loss/update remains finite and the
  skipped-update count is zero. The behavior below began before the link
  transition, so it is not a resume discontinuity.
- P_gradient_trend: rolling 500-update median `q_delta` gradient norm rises
  from 24.6 at updates 8,001..8,500 to 764 at 8,501..9,000, then remains 709,
  565, 399, and 228 across the successive windows through the current partial
  10,501+ window. The observed maximum is 1,353; one coupled estimator/global
  gradient spike reaches 5,374. Median total loss rises from roughly 6.46 to
  roughly 7.9 over the same period.
- controls: H keeps `q_delta` gradient medians near 9--10 and I near 4--7 in
  the same windows. I's estimator isolation remains effective apart from rare
  direct-path spikes. H pre/post-resume medians are effectively unchanged.
- interpretation: P's attached operator objective is producing a sustained
  coupled-gradient instability, consistent with the earlier direct-path
  degradation and distinct from an infrastructure or numerical failure. I is
  currently providing the intended causal control. This is descriptive
  evidence only: no arm is stopped, selected, or modified, and all continue
  unconditionally to the exact 32k endpoint and 1,200-episode evaluation.

### ANALYSIS_008001_011000 — 2026-08-22T08:16:14Z

- freeze_integrity: an independent read-only analysis froze exactly 3,000
  ordered rows per arm, updates 8,001..11,000, with no gaps, duplicates,
  nonfinite values, or skipped updates. Canonical row hashes are H
  `0dfeda4056d6f58cd5f9101211b09f32addd26b65cc33901052ec0f234f31f77`,
  P `cc15ea138cc2f233349d755c61904bff36780325fe47605b09c31d268fecf847`,
  and I `985c795be7e13d37da5d2cba3d85ca47f266be149fc6d28796956e92f3dc6764`.
  Every 500-step window has exactly 200 Long and 100 examples from each other
  suite; burn-in counts are also arm-identical. Sampling cannot explain the
  P-only instability.
- P_window_means_8k_to_11k: total loss is 6.546, 7.478, 8.030, 7.953,
  7.984, 8.112 across successive 500-step windows. Corresponding total/
  `q_delta` pre-clip gradients are 48.0/45.3, 674.2/673.5, 714.7/713.8,
  587.4/586.4, 428.9/427.9, and 366.3/352.2. Proposal gradient reaches
  20.7--29.6 in the unstable regime.
- controlled_geometry_8k_to_11k: mean total pre-clip gradient is H 9.77,
  P 469.93, I 5.80; `q_delta` is 9.53/466.53/5.52; estimator is
  0.073/3.377/0.142; proposal is 2.03/22.70/1.57. With global clipping at
  1.0, median clip scales are H 0.1021, P 0.00207, and I 0.1741. P therefore
  starves effective `q_action`, decoder, and proposal updates even though its
  dynamics graph never directly targets `q_action`.
- P_semantic_effect: mean live `q_delta` codes fall 17.36 -> 7.82 -> 6.52 ->
  5.70 -> 5.62 before only a small rebound to 6.66; spread falls 0.335 ->
  0.082. Effect gap returns near zero, and proposal/deployed overlap falls
  0.320 -> 0.236. Over the full interval P averages 8.28 live `q_delta`
  codes, spread 0.148, and deployed overlap 0.264 versus I 17.68/0.290/0.451.
  P's `q_action` remains broad (62.6 live, entropy 3.741), localizing failure
  to estimator--`q_delta` co-adaptation and the bridge to proposal rather than
  a direct `q_action` codebook collapse.
- deployed_cost: P proposal CE is 3.456 versus H 2.408 and I 2.753. P
  teacher/deployed decode is 0.0335/0.0760 with gap 0.0425, versus H
  0.0338/0.0575/0.0237 and I 0.0329/0.0593/0.0265. Lower P effect loss is
  therefore a shortcut signal, not evidence of executable operators.
- resume_control: H is unchanged across its 100-step pre/post-resume windows;
  I remains bounded. P does not stabilize: total gradient 269.6 -> 450.3,
  `q_delta` 268.4 -> 396.4, and estimator mean 0.344 -> 55.1, including an
  estimator spike of 5,373.5 at step 10,771. The instability began before
  resume and rebounds after it, so it is neither introduced nor cured by the
  link transition.
- interpretation: P versus I is a clean causal contrast. Allowing the operator
  objective into the online estimator creates a positive feedback loop,
  `q_delta` semantic collapse, and globally clipped branch starvation. I
  removes that feedback and remains bounded, but its weak effect model still
  requires the predeclared fixed-endpoint SR evaluation. No checkpoint or arm
  is selected from these metrics.

### OBSERVATION_016000 — 2026-08-22T09:54:13Z

- integrity: updates 15,501..16,000 are exact and contiguous for H/P/I, 500
  rows per arm, with zero skipped or nonfinite updates. Every lineage has
  `LATEST=16000`, exactly 80 nonzero retained shards, and W&B success through
  at least update 16,040 with zero failed health events. All train-02 jobs
  remain `RUNNING`.
- H: `q_delta` live/entropy/spread is 5/1.480/0.058 and remains collapsed;
  `q_action` is 57/3.693/0.560. MSE align is 0.612; proposal/deploy overlap
  0.508; proposal CE 2.451. Total/deployed/teacher decode is
  0.0375/0.0483/0.0259 with gap 0.0199. Legacy dynamics has cosine 0.939 but
  every delta-selection median is exactly zero. Estimator/bank/`q_delta`/
  proposal gradients are 0.0671/0.0622/7.467/2.088; global 7.823.
- P: `q_delta` 20/2.268/0.427 and `q_action` 64/3.786/0.585. Align CE is
  6.538, overlap 0.234, proposal CE 3.581, and deployed/teacher decode
  0.0724/0.0286 with a large 0.0413 gap. Effect/state/cosine are
  2.428/0.728/0.754; contrastive 3.538 and effect gap 0.0835. `delta_op`/
  aggregate selection are positive 0.00121/0.00100 across all horizons, so
  operator discrimination exists, but estimator/`q_delta`/proposal/global
  gradients are 72.62/27.46/11.31/81.96 and remain unstable.
- I: `q_delta` has contracted to 7/1.693/0.193 while `q_action` remains broad
  at 67/3.914/0.528. Align CE 4.852, overlap 0.445, proposal CE 2.828, and
  deployed/teacher decode 0.0481/0.0252 with gap 0.0211 keep the direct route
  close to H. The isolated operator remains near-null: state/effect/cosine
  2.070/2.406/0.187, contrastive at the random 4.426 value, effect gap near
  zero, and delta selection near zero. Estimator gradient is 0.0453 versus P
  72.62, with global 5.762.
- trend_8k_to_16k: H improves its executable surrogate while never forming a
  transition vocabulary. I successfully protects executable behavior but its
  isolated operator loses support from 20 to 7 codes and does not learn useful
  effects. P intermittently learns real effect separation, but the coupled
  optimization oscillates: `q_delta` gradient peaked around 645 in the 9--10k
  bin, then estimator coupling shifted from roughly 0.3 to 30--90 after 11k;
  direct overlap falls 0.328 -> 0.234 and the decode gap worsens.
- interpretation: the experiment has isolated a genuine tradeoff, not yet a
  winning method. H is the direct-policy control, P learns operator signal at
  the cost of severe feedback and deployed degradation, and I prevents that
  cost but currently also prevents meaningful operator fitting. All continue
  unchanged to fixed 32k and unconditional SR; this observation is not a
  checkpoint-selection rule.

### TRAIN_LINK_02_COMPLETE — 2026-08-22T11:46:44Z

- status: H/P/I train-02 jobs `32710460` / `32710608` / `32710737` all
  completed `0:0` at exact committed updates 21,054 / 21,291 / 21,260.
  Train-03 jobs `32710461` / `32710609` / `32710738` are `RUNNING` and use
  the same online W&B IDs in `resume=must` mode.
- resume_integrity: first appended updates are exactly H 21,055, P 21,292,
  and I 21,261. For every arm, the ledger remains exactly updates 1..latest
  with row count equal to latest, no gaps, duplicates, or rollback. Raw/cache
  assets were fully reverified after train-02 and before train-03.
- boundary_metrics: ordinary batch variation is observed rather than a resume
  discontinuity. H global/`q_delta` gradient moves 6.43/6.11 -> 7.06/6.20;
  P global/proposal 27.07/27.06 -> 24.80/24.70; I global/`q_delta`
  4.13/3.85 -> 4.44/4.11.
- interpretation: the second authenticated handoff is exact and scientific
  differences persist independently of job boundaries. All three fixed
  lineages continue toward 32k without selection or thresholding.

### OBSERVATION_024000 — 2026-08-22T12:47:13Z

- integrity: updates 23,501..24,000 are exact and contiguous for H/P/I, 500
  rows per arm, with zero skips. Every arm has `LATEST=24000`, 96 nonzero
  retained shards, continuous row count equal to step, and W&B success beyond
  24k with zero failed events. All train-03 jobs remain `RUNNING`.
- H: `q_delta` live/entropy/spread remains 6/1.542/0.098 while `q_action` is
  58/3.711/0.569. MSE align is 0.625, proposal/deploy overlap 0.563, proposal
  CE 2.168, total/deployed/teacher decode 0.0281/0.0357/0.0202, and coefficient
  L2 0.243. Legacy dynamics cosine is 0.941 but effect/contrastive/delta
  selection remain absent or zero. Estimator/bank/`q_delta`/proposal/global
  gradients are 0.0828/0.0712/6.025/2.957/6.773.
- P: `q_delta` 14/1.623/0.571 and `q_action` 65/3.821/0.574. Align CE worsens
  to 8.551, overlap is 0.297, proposal CE 3.293, and deployed/teacher decode
  0.0543/0.0214 with gap 0.0288. Operator fitting is now strong: effect 0.670,
  state 0.0324/cosine 0.992, contrastive 1.668/top1 0.938, effect gap 0.716,
  and `delta_op`/aggregate selection 0.00949/0.00907 with every horizon
  0.0081--0.0088. Estimator/`q_delta` gradients settle to 0.858/1.109, but
  proposal gradient dominates at 26.14 and global is 26.27.
- I: `q_delta` partially recovers to 15/2.150/0.285 and `q_action` remains
  broad at 70/3.974/0.525. Align CE 4.841, overlap 0.461, proposal CE 2.804,
  and total/deployed/teacher decode 0.0304/0.0388/0.0209 keep the direct route
  healthy. The isolated operator is still near chance: state/effect/cosine
  1.802/2.482/0.273, contrastive 4.424, effect gap 0.000113, and selection near
  4.7e-5. Estimator isolation holds at gradient 0.0757; global is 4.355.
- trend_16k_to_24k: H's executable route improves strongly while its
  transition code stays collapsed. I improves direct metrics and modestly
  reopens `q_delta`, but still fails to fit useful effects. P moves from an
  extreme-gradient regime into a powerful operator-fit phase, yet pays with
  worse semantic alignment, low direct overlap, and proposal-dominated
  gradients. This is an oscillatory operator/direct-path tradeoff, not clean
  convergence.
- interpretation: none of the three is a surrogate-metric winner across both
  method and deployment. H remains the direct-policy anchor, P demonstrates
  that attached operator learning is possible but interferes with execution,
  and I demonstrates protection without adequate operator learning. All
  continue unchanged to exact 32k and unconditional SR.

### FIXED_TRAINING_ENDPOINT_032000 — 2026-08-22T15:43:05Z

- train03_transition: H/P/I train-03 jobs `32710461` / `32710609` /
  `32710738` completed `0:0` at exact committed updates 31,493 / 31,843 /
  31,743. Train-04 first appended updates are exactly 31,494 / 31,844 /
  31,744, with no gap, duplicate, or rollback.
- fixed_endpoint: train-04 jobs `32710462` / `32710610` / `32710739` all
  completed `0:0` after reaching exact update 32,000. Every metrics ledger is
  exactly 32,000 ordered rows for updates 1..32,000; `LATEST=32000`; all 16
  endpoint rank shards exist per arm. W&B has an `ok=true` event at update
  32,000 for H/P/I, with zero W&B failures, skipped updates, or nonfinite
  updates over each full run.
- later_links: train-05/train-06 are authenticated exact-endpoint no-ops only;
  they perform no optimizer updates. P has completed consolidation and begun
  all three unconditional eval seeds; I is consolidating; H is completing its
  endpoint no-op/verification chain. No outcome or scientific gate controls
  these transitions.
- interpretation: all three fresh methods have now completed the exact same
  fixed training budget. The remaining authority is checkpoint integrity plus
  the unconditional 3 x 400 episode evaluation and exact 1,200-row merge.

### ANALYSIS_TERMINAL_030001_032000 — 2026-08-22T15:43:05Z

- freeze: exact steps 30,001..32,000, 2,000 ordered rows per arm. Values below
  are medians; every arm has 112 nonzero retained shards and an authenticated
  terminal checkpoint.
- H: `q_delta` live/entropy/spread is 7/1.669/0.148 versus `q_action`
  59/3.718/0.570. MSE align 0.627, deployed overlap 0.633, proposal CE 1.910,
  deployed/teacher decode 0.0287/0.0188, gap 0.00891, and coefficient L2
  0.179 make it the strongest direct surrogate. Legacy state/cosine is
  0.322/0.940, but all operator effect/contrastive/delta metrics are zero.
  Estimator/bank/`q_delta`/proposal/global gradients are
  0.188/0.068/5.705/4.315/7.186.
- P: `q_delta` 18/1.873/0.572 and `q_action` 66/3.835/0.573. Align CE remains
  very poor at 7.866, overlap 0.336, proposal CE 3.125, and deployed/teacher
  decode 0.0486/0.0196 with gap 0.0263 and coefficient L2 0.427. The operator
  is genuinely discriminative: state/effect/cosine 0.094/0.737/0.971,
  contrastive 1.335/top1 0.969, effect gap 0.663, and `delta_op`/selection
  0.01463/0.01454. Estimator/`q_delta` gradients settle to 0.705/0.808, but
  proposal dominates at median 23.742 [p10 11.847, p90 53.295], with global
  23.811.
- I: `q_delta` 17/2.368/0.331 and `q_action` 71/3.996/0.527. Align CE 4.852,
  overlap 0.500, proposal CE 2.614, deployed/teacher decode 0.0321/0.0197,
  gap 0.0112, and coefficient L2 0.239 preserve a substantially healthier
  direct path than P while retaining more transition support than H. The
  isolated operator remains chance-level: state/effect/cosine
  1.808/2.502/0.218, contrastive 4.425, effect gap about zero, and deltas near
  2--3e-5. Estimator/`q_delta`/proposal/global gradients are
  0.118/3.143/3.522/4.787.
- terminal_hypotheses: H should be the most likely direct-SR winner based on
  its proposal/decode bridge but will not validate the operator method. P is
  the only successful operator learner but risks low direct SR from severe
  alignment and proposal mismatch. I is the protected compromise and could
  approach H's SR, but its operator contribution is not deployable. These are
  preregistered interpretations only; the exact 1,200-episode results remain
  authoritative and are run for every arm without thresholding.

### FINAL_EVALUATION_RESULTS — 2026-08-22T16:11:34Z

- execution: all 33 version-2 jobs completed `0:0`: H `32710459..32710469`,
  P `32710607..32710617`, and I `32710736..32710746`. Each arm trained for
  exactly 32,000 contiguous updates, consolidated its fixed endpoint, ran
  seeds 0/1/2 for exactly 400 episodes each, and merged exactly 1,200 unique
  episodes. There were zero training skips, nonfinite updates, W&B failures,
  evaluation errors, threshold decisions, checkpoint-selection decisions, or
  outcome gates. Every failed episode ran to the fixed 512-step cap.
- exact_results:

  | Arm | Successes / episodes | End-to-end SR | Seeds 0/1/2 | Spatial | Object | Goal | Long | Capped failures |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | H | 426 / 1,200 | 35.500% | 141 / 148 / 137 | 37.667% | 45.667% | 53.333% | 5.333% | 774 |
  | P | 405 / 1,200 | 33.750% | 137 / 126 / 142 | 42.667% | 34.333% | 46.000% | 12.000% | 795 |
  | I | 508 / 1,200 | **42.333%** | 177 / 162 / 169 | 48.333% | 45.667% | 58.667% | 16.667% | 692 |
  | prior dual-code | 550 / 1,200 | **45.833%** | 178 / 180 / 192 | 54.000% | 45.000% | 71.333% | 13.000% | 650 |
  | frozen baseline | 447 / 1,200 | 37.250% | 149 / 149 / 149 | 40.667% | 29.667% | 53.667% | 25.000% | 753 |

- raw_deltas: I is +82 successes / +6.833 pp over H and +103 / +8.583 pp
  over P. It is +61 / +5.083 pp over the frozen baseline, but remains -42 /
  -3.500 pp below the prior dual-code best. H is -1.750 pp versus baseline and
  -10.333 pp versus prior; P is -3.500 pp versus baseline and -12.083 pp
  versus prior. The prior 45.833% checkpoint therefore remains the highest
  measured end-to-end policy.
- paired_inference: all five methods use exactly the same 1,200 episode keys,
  `env_seed`, and `policy_seed`. The fixed suite-stratified 10,000-draw task
  bootstrap uses seed 49,666 and matrix SHA
  `1e570b6d13426c8fbd58016d0fba6869dc18aa3151dfdbc0bab357373cacf32e`.

  | Comparison | Delta | Discordant A-only/B-only | Exact episode p | Task +/−/tie | Task-sign p | Task-bootstrap 95% CI |
  |---|---:|---:|---:|---:|---:|---:|
  | I - P | +8.583 pp | 283 / 180 | 1.95e-6 | 23 / 8 / 9 | 0.01067 | [+3.083, +14.000] pp |
  | I - H | +6.833 pp | 258 / 176 | 9.67e-5 | 26 / 13 / 1 | 0.0533 | [+1.750, +11.750] pp |
  | H - P | +1.750 pp | 214 / 193 | 0.3215 | 19 / 18 / 3 | 1.000 | [-4.833, +8.250] pp |
  | I - baseline | +5.083 pp | 269 / 208 | 0.005953 | 20 / 17 / 3 | 0.7428 | [-0.417, +10.667] pp |
  | I - prior | -3.500 pp | 197 / 239 | 0.04946 | 15 / 22 / 3 | 0.3240 | [-8.083, +0.833] pp |

- statistical_read: I's improvement over P is broad and survives the grouped
  task bootstrap; I beats P in every seed and every suite. I's numerical gain
  over baseline is not a task-level decisive win because its grouped CI
  slightly crosses zero. Likewise, the experiment does not establish that I
  differs from the prior best at task level, although the prior has the higher
  observed SR. These statements distinguish the paired episode result from
  the more conservative 40-task uncertainty.
- causal_result:
  - P versus I is the cleanest controlled contrast. Their intended method
    difference is operator-objective gradient isolation from the online
    estimator. Isolation adds 103 successes and +8.583 pp, supporting it as
    the correct direction within this protocol.
  - P learns the strongest operator surrogate (`effect_gap` 0.663,
    contrastive top-1 0.969, positive selection about 0.0145), but scores only
    33.75%. Its poor terminal alignment CE 7.866, proposal overlap 0.336,
    proposal-dominated gradients, and deployed decode 0.0486 show that
    attached operator learning damages the route actually used by direct
    inference. Better bank metrics do not compensate because direct R0
    inference does not use bank rollouts.
  - I protects the deployed path and achieves the best new-arm SR while its
    isolated operator remains near chance. This proves that protecting action
    realizability matters more than optimizing the disconnected operator
    surrogate, but it does not yet deliver the full LOOM method.
  - H does not reproduce the prior 45.833% result under the common changed
    sampling/history protocol. Its strongest direct surrogates yield only
    35.50%, dominated by Long at 5.33%. Therefore the 20/20/20/40 suite mix
    plus randomized recurrent prefixes is not a free improvement and should
    not replace the exact prior-best data/history policy without a controlled
    ablation.
- immutable_authority:
  - source closure:
    `690e018045a964790507307ee8125181324c0f3b6d6a717ca304aa6cb0a76836`
    over 59/59 files at commit
    `2e63f551a0fc38495e99d9715ebd134cddca90a1`;
  - H checkpoint/result/merge receipt:
    `3117da3e22d7efac0a7de5818cab49d71b1f81e9c0428a0789717c1d75720a16` /
    `7113f7ecdb40e3ea9f903c2bfb6b135e33e3276ce83125c7419d28f54e009ff7` /
    `e2dfef9ddc11ce0302e7f309e507e7215e9bd2f42ba8c226125d067ed7972bf4`;
  - P checkpoint/result/merge receipt:
    `aa21a0c19a3d6e0b88700138ec919f1af8d45d88eb52821c9038956905264bdc` /
    `8a5b1e754e6a1e92d617e87974c46d9ddef7567446a0e16afca9f5eaf8c0f52a` /
    `798f28f0ba3eb728f887754c350203ab88b6df2f09e8fd2134c090de3383c3de`;
  - I checkpoint/result/merge receipt:
    `c3767f162a65724b5e3b78ebbfd71884559e0089af2aad6725f449ce87c20b6b` /
    `917115ba66ca0d5985f95ef4330723e39a06c1e69810c278c6e6cf5d8a707c6a` /
    `f26d9cef35cd1d1cb5809ad4da33969db13029d06e7e407f1caff63728da3c54`.
- W&B: final online training runs are H
  `https://wandb.ai/crlc112358/loom-r0-protected-arms/runs/bb865fe0fa68442f`,
  P `.../a1d8a5066557422d`, and I `.../7ad4e3ffe04842a4`. Each contains the
  exact 1,600 health acknowledgements at 20-update cadence through 32,000,
  all successful.
- independent_audit: a separate read-only replay found no integrity blocker,
  reproduced all exact counts, stored baseline comparisons, pairwise task
  rows, bootstrap intervals, and artifact hashes, and confirmed that each
  merged payload equals the disjoint union of its three singleton results.
- next_direction: retain the prior 45.833% policy and its exact original
  sampler/history settings as the action-realizability anchor. Train the
  operator branch from `q_delta` with the estimator and `q_action` frozen or
  stop-gradient protected, using per-module gradient normalization/clipping;
  do not let operator loss enter the direct action route. First verify that
  this isolated branch can learn effect separation without reducing the
  anchor's fixed-checkpoint direct SR. Only then connect it at inference as a
  reranker: proposal generates candidate code sequences, bank rollouts score
  effects, and a goal-conditioned scorer selects among candidates. The next
  evaluation should again publish the same exact 3 x 400 protocol
  unconditionally, with diagnostics used only for explanation.
