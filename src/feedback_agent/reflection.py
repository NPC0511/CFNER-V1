"""Task-end reflection storage for risk-prediction auditing."""

import json
import os

from .state import TaskReflection


class ReflectionMemory(object):
    """Persist task summaries without granting any training authority."""

    def __init__(self, output_dir, enabled=True, forgetting_threshold=1.0):
        self.output_dir = output_dir
        self.enabled = bool(enabled)
        self.forgetting_threshold = float(forgetting_threshold)

    def write(self, state, semantic_risk=None, action_records=None):
        if not self.enabled or state is None:
            return ""
        old_states = state.old_classes
        observed_forgetting = {
            name: item.f1_drop for name, item in old_states.items()
            if item.f1_before is not None and item.f1_current is not None
        }
        risk_nodes = (semantic_risk or {}).get("nodes", {})
        confirmed, missed = [], []
        for name, f1_drop in observed_forgetting.items():
            node = risk_nodes.get(name, {})
            predicted_high = node.get("level") in ("medium", "high")
            forgotten = f1_drop >= self.forgetting_threshold
            if predicted_high and forgotten:
                confirmed.append(name)
            elif forgotten and not predicted_high:
                missed.append(name)
        reflection = TaskReflection(
            task_id=state.task_id,
            summary_tags=(["forgetting_observed"] if observed_forgetting
                          else ["no_old_class_evaluation"]),
            next_task_focus=sorted(set(confirmed + missed)),
            metrics={
                "new_types": list(state.new_types),
                "old_types": list(state.old_types),
                "f1_before": {name: item.f1_before for name, item in old_states.items()},
                "f1_current": {name: item.f1_current for name, item in old_states.items()},
                "f1_drop": observed_forgetting,
                "feature_drift": {name: item.feature_drift for name, item in old_states.items()},
                "prototype_similarity": {name: item.prototype_similarity for name, item in old_states.items()},
                "predicted_risk": risk_nodes,
                "confirmed_risks": confirmed,
                "missed_risks": missed,
            },
            action_records=list(action_records or []),
        )
        directory = os.path.join(self.output_dir, "feedback_monitor", "reflection")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "%s_task_%d.json" % (state.domain, state.task_id))
        latest_path = os.path.join(directory, "%s_latest.json" % state.domain)
        for destination in (path, latest_path):
            with open(destination, "w", encoding="utf-8") as handle:
                json.dump(reflection.to_dict(), handle, ensure_ascii=True, indent=2)
        return path
