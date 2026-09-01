"""Bounded action specifications for later Feedback-RDP interventions."""

from typing import Dict, List

from .state import ActionRecord, ActionRequest, TrainingState


ALLOWED_ACTIONS = {
    "no_action",
    "increase_old_distillation",
    "increase_prototype_anchor",
    "reduce_pseudo_label_weight",
}


class ObserveOnlyPolicy(object):
    """Validates proposed actions and records that no mutation was applied."""

    def __init__(self, max_action_delta=0.05, cooldown_steps=200):
        self.max_action_delta = float(max_action_delta)
        self.cooldown_steps = int(cooldown_steps)

    def validate(self, request: ActionRequest) -> ActionRecord:
        if request.action not in ALLOWED_ACTIONS:
            return ActionRecord(request, False, "action_not_allowlisted")
        if request.action != "no_action" and not request.targets:
            return ActionRecord(request, False, "missing_target_classes")
        if abs(float(request.delta)) > self.max_action_delta:
            return ActionRecord(request, False, "delta_exceeds_limit")
        return ActionRecord(
            request, True, "observe_only_no_training_mutation",
            metrics_before=dict(request.trigger_state)
        )

    def diagnose(self, state: TrainingState) -> ActionRequest:
        """Create a deterministic proposal without mutating training state."""
        targets = sorted(name for name, item in state.old_classes.items()
                         if item.f1_drop > 0 or item.feature_drift > 0)
        return ActionRequest("no_action", state.step, targets=targets,
                             trigger_state=state.to_dict())


class AdaptiveDistillationPolicy(ObserveOnlyPolicy):
    """Small, temporary KD boosts driven by observed old-class drift."""

    def __init__(self, drift_threshold=0.15, trigger_patience=3,
                 max_action_delta=0.05, max_multiplier=1.5,
                 effect_window_steps=500, cooldown_steps=200):
        super().__init__(max_action_delta=max_action_delta, cooldown_steps=cooldown_steps)
        self.drift_threshold = float(drift_threshold)
        self.trigger_patience = max(int(trigger_patience), 1)
        self.max_multiplier = max(float(max_multiplier), 1.0)
        self.effect_window_steps = max(int(effect_window_steps), 1)
        self.streak = 0
        self.active_until = None
        self.cooldown_until = 0
        self.multiplier = 1.0

    def update(self, step, drift):
        """Return one action record, or None, without mutating the trainer."""
        if self.active_until is not None and step >= self.active_until:
            previous = self.multiplier
            self.multiplier = 1.0
            self.active_until = None
            self.cooldown_until = step + self.cooldown_steps
            return ActionRecord(
                ActionRequest("no_action", step, delta=0.0,
                              trigger_state={"event": "effect_window_complete",
                                             "previous_multiplier": previous}),
                True, "rollback_to_baseline_multiplier", rollback=True
            )

        targets = sorted(name for name, value in drift.items()
                         if value >= self.drift_threshold)
        self.streak = self.streak + 1 if targets else 0
        if (self.active_until is not None or step < self.cooldown_until
                or self.streak < self.trigger_patience or not targets
                or self.multiplier >= self.max_multiplier):
            return None

        delta = min(self.max_action_delta, self.max_multiplier - self.multiplier)
        request = ActionRequest(
            "increase_old_distillation", step, targets=targets, delta=delta,
            duration_steps=self.effect_window_steps, cooldown_steps=self.cooldown_steps,
            trigger_state={"feature_drift": dict(drift), "trigger_streak": self.streak,
                           "drift_threshold": self.drift_threshold}
        )
        record = self.validate(request)
        if record.accepted:
            self.multiplier += delta
            self.active_until = step + self.effect_window_steps
            record.reason = "bounded_distillation_boost_applied"
            record.metrics_after = {"distillation_multiplier": self.multiplier,
                                    "active_until_step": self.active_until}
        return record


class RiskGatedDistillationPolicy(AdaptiveDistillationPolicy):
    """Apply the bounded KD action only to classes selected by a risk map."""

    LEVELS = {"low": 0, "medium": 1, "high": 2}

    def __init__(self, min_risk_level="medium", **kwargs):
        super().__init__(**kwargs)
        if min_risk_level not in self.LEVELS:
            raise ValueError("Invalid minimum risk level: %s" % min_risk_level)
        self.min_risk_level = min_risk_level

    def update(self, step, drift, risk_map):
        """Filter drift candidates through the deterministic semantic risk map."""
        nodes = getattr(risk_map, "nodes", {}) if risk_map is not None else {}
        edges = getattr(risk_map, "edges", []) if risk_map is not None else []
        minimum = self.LEVELS[self.min_risk_level]
        edge_targets = {edge.target for edge in edges
                        if self.LEVELS.get("high" if edge.risk >= 0.60
                                           else "medium" if edge.risk >= 0.30
                                           else "low", 0) >= minimum}
        eligible = {
            name: value for name, value in (drift or {}).items()
            if ((edge_targets and name in edge_targets)
                or (not edge_targets and name in nodes
                    and self.LEVELS.get(nodes[name].level, 0) >= minimum))
        }
        record = super().update(step, eligible)
        if record is not None:
            record.request.trigger_state["risk_controller"] = {
                "minimum_level": self.min_risk_level,
                "eligible_targets": sorted(eligible),
                "risk_nodes": {name: nodes[name].to_dict() for name in eligible},
                "risk_edges": [edge.to_dict() for edge in edges
                                if edge.target in eligible],
            }
        return record
