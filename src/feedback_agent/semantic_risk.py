"""Deterministic semantic-risk aggregation for observed class failures.

This layer only summarizes evidence.  It has no training authority; Qwen can
later contribute additional reason tags, while the controller remains the
source of executable actions.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class RiskNode:
    entity_type: str
    score: float = 0.0
    level: str = "low"
    reasons: List[str] = field(default_factory=list)
    evidence: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class SemanticRiskMap:
    task_id: int
    step: int
    nodes: Dict[str, RiskNode] = field(default_factory=dict)
    source: str = "python_rules"

    def to_dict(self):
        return {"task_id": self.task_id, "step": self.step,
                "source": self.source,
                "nodes": {name: node.to_dict() for name, node in self.nodes.items()}}


def build_risk_map(state, drift_threshold=0.15, confusion_threshold=0.20,
                   entropy_threshold=1.0, similarity_threshold=0.50,
                   high_score=0.60, medium_score=0.30):
    """Build a bounded [0, 1] risk score from current monitor evidence."""
    risk_map = SemanticRiskMap(task_id=state.task_id, step=state.step)
    for entity_type, item in state.old_classes.items():
        reasons = []
        evidence = {}
        signals = []
        if item.feature_drift >= drift_threshold:
            reasons.append("feature_drift")
            signals.append(min(item.feature_drift / max(drift_threshold, 1e-8), 1.0))
            evidence["feature_drift"] = item.feature_drift
        if item.teacher_confusion is not None and item.teacher_confusion >= confusion_threshold:
            reasons.append("teacher_confusion")
            signals.append(min(item.teacher_confusion / max(confusion_threshold, 1e-8), 1.0))
            evidence["teacher_confusion"] = item.teacher_confusion
        if item.pseudo_entropy is not None and item.pseudo_entropy >= entropy_threshold:
            reasons.append("pseudo_uncertainty")
            signals.append(min(item.pseudo_entropy / max(entropy_threshold, 1e-8), 1.0))
            evidence["pseudo_entropy"] = item.pseudo_entropy
        if (item.prototype_similarity is not None
                and item.prototype_similarity <= similarity_threshold):
            reasons.append("prototype_instability")
            signals.append(min((similarity_threshold - item.prototype_similarity)
                               / max(similarity_threshold, 1e-8), 1.0))
            evidence["prototype_similarity"] = item.prototype_similarity
        if item.f1_drop > 0:
            reasons.append("f1_drop")
            signals.append(min(item.f1_drop, 1.0))
            evidence["f1_drop"] = item.f1_drop
        score = min(1.0, sum(signals) / max(len(signals), 1))
        level = "high" if score >= high_score else "medium" if score >= medium_score else "low"
        risk_map.nodes[entity_type] = RiskNode(entity_type, score, level, reasons, evidence)
    return risk_map
