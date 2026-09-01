# Selective Risk Graph Migration Plan

## Purpose

The current project has monitoring, reflection, directed risk edges, semantic
memory, and task-level Qwen plumbing. It should stop adding unrelated signals
and turn the graph into one precise, testable training mechanism.

The reference archive is useful as a design reference, but must not be copied
wholesale: it mixes several losses, calibrators, LLM utilities, and known
defects. The first question is:

```text
Can a directed new_type -> old_type risk graph reduce old-class forgetting by
applying stronger KD only to old labels predicted to be at risk?
```

## Current Assets

- Sampled monitor, JSONL, task summaries, F1 baselines, and reflection.
- Directed `new_type -> old_type` risk edges and graph cache.
- Reviewed CoNLL2003 semantic memory.
- Task-level and edge-level Qwen invocation and cache.
- A bounded global KD multiplier controller.

The global multiplier is not the desired main mechanism: it applies one global
boost while the graph identifies directed class pairs.

## Selective Migration

### Migrate Now

| Reference concept | Current target |
|---|---|
| Serializable task RiskGraph | Stabilize the current edge-cache schema. |
| Rule/LLM decomposition | Store rule, LLM, observed, historical, and final risk separately. |
| `risk_edges -> old_label_weights` | First graph-driven training intervention. |
| Frozen per-task policy | Replace online KD pulses with a fixed task policy. |
| Risk-vs-forgetting analysis | Validate graph prediction before making claims. |
| Next-task reflection evidence | Add after weighted KD is stable. |

### Defer

| Reference feature | Reason |
|---|---|
| Prototype feature anchor | Reference version may anchor known new-class tokens to old prototypes. |
| Contrastive loss | A second loss hides whether weighted KD works. |
| Pseudo-label routing/verifier | Needs token-level validation and independent ablation. |
| Calibrator and many fusion knobs | Insufficient history; risks becoming a tuning loop. |
| Multiple actions per task | Prevents causal attribution. |

## Replace the Global Controller

Do not use this as the main method:

```text
drift threshold -> global KD +0.05 -> temporary window -> rollback -> repeat
```

Use this instead:

```text
task-start directed graph
-> aggregate edge risk by target old type
-> map target type to B-/I- label weights
-> freeze weights for this task
-> use weighted logits KD during the task
-> task-end reflection evaluates graph and policy
```

The graph decides which labels receive protection. A declared fixed profile,
not an online search, decides the bounded level of protection.

## Implementation Order

### A. Freeze Observation Baseline

For graph experiments disable the current pulse controller:

```yaml
agent_mode: observe_only
risk_controller_enabled: false
```

Keep Qwen, graph cache, and monitor records. Reject Qwen template echoes such
as all-zero components plus placeholder tags.

Acceptance: no graph experiment silently skips training due to an old
checkpoint; Qwen cache distinguishes `ok`, `partial`, `fallback`, and
`template_rejected`.

### B. Stabilize RiskEdge Schema

Every edge must record:

```json
{
  "source": "organisation",
  "target": "location",
  "rule_risk": 0.85,
  "llm_risk": 0.0,
  "observed_risk": 0.0,
  "historical_vulnerability": 0.0,
  "final_risk": 0.85
}
```

Do not collapse components before the final risk. Historical vulnerability is
target-level evidence, not a source-target causal claim.

### C. Task-Start Risk-Weighted KD

Implement only one action:

```text
edge final risk -> target old type -> B/I label weights -> weighted logits KD
```

Freeze a monotonic profile before running an ablation:

```text
risk < 0.50         -> 1.00
0.50 <= risk < 0.75 -> 1.15
risk >= 0.75        -> 1.30
```

The mapping is computed once per task and saved in its risk graph. Required
trainer diagnostics are per-label KD weight, valid KD token counts, unweighted
KD loss, weighted KD loss, and total weighted contribution.

Acceptance: high-risk labels have higher KD contribution, non-risk labels stay
at baseline, and new-class CE is unchanged.

### D. Single-Action Ablation

After Phase C is verified:

```text
A. RDP baseline
B. RDP + monitor + graph logging
C. RDP + rule risk-weighted KD
D. RDP + rule + Qwen risk-weighted KD
```

Freeze seed, order, epochs, checkpoint rule, and baseline hyperparameters.
Report old/new/seen F1, forgetting, per-edge target F1 drop, Qwen status, and
non-template Qwen edge count.

### E. Reflection Update

Only after weighted KD is stable, write one risk-vs-forgetting row per edge and
use bounded historical evidence for the next task. A reflection must never
modify the task that produced it.

### F. Additional Actions

Only after a verified weighted-KD effect:

1. Correctly masked prototype anchor.
2. Risk-aware pseudo-label routing.
3. Contrastive separation for selected pairs.
4. Gradient intervention after conflict is validated against forgetting.

Each action needs an individual ablation before combinations.

## Qwen’s Role

Qwen provides semantic prior evidence only. Python validates it, calculates
numeric risk, freezes the task policy, and executes loss changes. A Qwen edge
with all-zero components and a prompt placeholder reason is template output,
not semantic analysis; reject it and retain the reviewed rule prior.

## Definition of Progress

```text
Qwen: non-template validated edge priors.
Graph: traceable component-to-final-risk records.
Weighted KD: label-specific loss contribution changes.
Method: better old-class protection without unacceptable new-class loss.
```

Until Phase C is complete, the graph is an analysis artifact, and F1 gains are
not an expected result.
