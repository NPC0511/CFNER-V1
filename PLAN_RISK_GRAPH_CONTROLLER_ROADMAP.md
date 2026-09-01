# Risk Graph and Controller Roadmap

## 1. Design Position

The controller is not intended to search for the best hyperparameters. Its job
is to convert observed evidence into a bounded, auditable intervention. The
research question is whether risk-aware intervention reduces forgetting while
preserving new-class learning, not whether a large collection of thresholds
can be tuned until one run improves.

The separation is:

```text
semantic prior / Qwen explanation -> risk graph
training measurements            -> evidence
deterministic controller         -> bounded action
task-end reflection              -> evidence for the next task
```

Qwen may explain or rank candidate risks, but it must not emit arbitrary loss
weights or execute training actions.

## 2. Current Baseline

Completed in the current repository:

- RDP continual NER training and cumulative checkpoint selection.
- Observe-only monitor with sampled JSONL and task summaries.
- Feature drift, prototype statistics, prototype similarity.
- Teacher confusion, pseudo-label confidence/entropy/coverage.
- Gradient conflict observation.
- Structured training state and task-end reflection memory.
- Rule-based risk nodes and a risk-gated KD controller.

The current controller is deliberately narrow: it can only increase the
existing distillation multiplier. The current risk map is still node-based;
the target graph is directed `new_type -> old_type`.

## 3. Stage-by-Stage Plan

### C0: Freeze Interfaces and Reproducibility

Code:

- Freeze `TrainingState`, `SemanticRiskMap`, `ActionRequest`, and
  `ActionRecord` schemas.
- Add a configuration snapshot and code commit id to every task summary.
- Keep `observe_only` as an explicit control condition.

Acceptance:

- The same seed and configuration produce the same risk/action records.
- No controller action is possible in `observe_only` mode.

### C1: Correct Directed Risk Graph

Code:

- Add `RiskEdge(source_new_type, target_old_type, risk, reasons, evidence)`.
- Build one edge for every current `new_type -> old_type` pair.
- Keep rule risk, model evidence, historical risk, and LLM risk as separate
  fields; compute the fused risk in Python.
- Persist task graph and latest graph under the experiment directory.

Acceptance:

- A task with one new class and three old classes produces exactly three
  directed edges.
- Every edge can be traced to its rule/evidence source.

### C2: Evidence Fusion Without Training Mutation

Code:

- Aggregate sampled teacher confusion, drift, prototype similarity, entropy,
  coverage, and reflection evidence by target old class.
- Map target-level evidence onto the relevant directed edges.
- Keep semantic prior and observed training evidence separately visible.

Acceptance:

- Risk logs show prior risk, observed evidence, and fused risk separately.
- This stage does not modify any loss or optimizer state.

### C3: Minimal Controller Contract

Controller inputs:

- Current directed risk edges.
- Sampled training evidence.
- Explicit safety limits.

Controller output:

- `no_action`, or one `increase_old_distillation` request.

Decision rule:

```text
high/medium edge risk
AND repeated evidence at observation points
AND target has enough samples
AND no active action/cooldown
=> bounded KD request
```

The controller must not optimize thresholds online. Thresholds are frozen for
an ablation and changed only between experiment batches.

Acceptance:

- One observation point can create at most one action.
- Every action has a target edge, delta, duration, cooldown, and rollback rule.
- All rejected decisions have an explicit reason.

### C4: Action Effect Window and Rollback

Code:

- Record target old-class F1/dev proxy, drift, similarity, conflict, and new
  class F1 immediately before an action.
- Aggregate the same metrics over the effect window.
- Roll back when the action expires or violates a new-class/cumulative-F1
  guardrail.
- Store `metrics_before`, `metrics_after`, `accepted`, and `rollback`.

Acceptance:

- An action can be audited from one JSON record and its effect-window records.
- A failed action returns to the previous multiplier.

### C5: Reflection and Historical Graph Update

Code:

- At task end, compare predicted high-risk edges with observed target F1 drop
  and new-to-old confusion.
- Mark edges as confirmed, missed, or unsupported.
- Use bounded historical evidence to update the next task's edge risk.
- Never use the current task's final test result to alter that task's training.

Acceptance:

- The next task can load the previous reflection.
- Historical updates are bounded and separately logged from semantic priors.

### C6: Add One More Action at a Time

Only after C3-C5 are stable:

1. Risk-weighted prototype anchor.
2. Risk-aware pseudo-label threshold.
3. Risk-guided contrastive separation.
4. Gradient projection or clipping, only after conflict evidence is validated.

Each action gets its own flag, safety limit, effect window, rollback rule, and
single-action ablation. Do not enable multiple new actions in the first test.

### C7: Qwen Semantic Assistant

Qwen is added after the rule/controller interface is frozen.

Input:

- New and old entity definitions.
- Annotation rules and examples.
- Directed rule graph.
- Previous reflection memory.

Output:

- Discrete semantic components.
- Reason tags.
- Candidate risk edges.
- Optional focus classes.

Python validates the output, calculates numeric risk, and decides actions.
Qwen failure falls back to rules and does not stop training.

Acceptance:

- Qwen is called at task level, not every batch.
- `Qwen-analysis-only` is independently measurable.
- Qwen cannot change arbitrary numerical loss coefficients.

### C8: Formal Ablation and Generalization

Freeze all rules and thresholds before formal evaluation. Compare:

- RDP baseline.
- RDP + monitor.
- RDP + rule risk graph.
- RDP + Python controller.
- RDP + Qwen analysis only.
- RDP + controller + Qwen suggestions.

Use multiple class orders, seeds, and datasets. Report new-class F1, old-class
F1, seen-class F1, average forgetting, action frequency, rollback frequency,
and risk-vs-forgetting correlation.

## 4. Avoiding a Tuning Project

The controller should expose only safety and protocol parameters:

- observation interval;
- minimum evidence count/patience;
- maximum delta;
- maximum multiplier;
- effect-window length;
- cooldown;
- guardrail thresholds.

These are not independently optimized per task. Use a small development set,
freeze one configuration per ablation, and report all values. Do not tune on
test F1, do not let the controller search over actions, and do not introduce a
new threshold unless it corresponds to a stated failure mode.

The strongest evidence is not a single improved score. It is:

```text
predicted high-risk edges have larger observed forgetting
AND bounded actions improve old-class protection
WITHOUT unacceptable new-class degradation
AND the result survives seeds, orders, and datasets.
```

## 5. Immediate Next Step

Implement C1 only: directed `new_type -> old_type` edges and per-edge JSON
cache. Keep the current node-based risk map as a compatibility view. Do not
add prototype-anchor or pseudo-label actions until the directed graph and
effect-window records are validated.
