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
class RiskEdge:
    """Directed interference risk from a current new type to an old type."""
    source: str
    target: str
    risk: float = 0.0
    risk_type: List[str] = field(default_factory=list)
    reason: str = ""
    evidence: Dict[str, Optional[float]] = field(default_factory=dict)
    source_kind: str = "observed_training_evidence"

    def to_dict(self):
        return asdict(self)


@dataclass
class SemanticRiskMap:
    task_id: int
    step: int
    nodes: Dict[str, RiskNode] = field(default_factory=dict)
    edges: List[RiskEdge] = field(default_factory=list)
    source: str = "python_rules"

    def to_dict(self):
        return {"task_id": self.task_id, "step": self.step,
                "source": self.source,
                "nodes": {name: node.to_dict() for name, node in self.nodes.items()},
                "edges": [edge.to_dict() for edge in self.edges]}


def build_risk_map(state, drift_threshold=0.15, confusion_threshold=0.20,
                   entropy_threshold=1.0, similarity_threshold=0.50,
                   high_score=0.60, medium_score=0.30,
                   semantic_memory=None, qwen_assessment=None,
                   rule_risk_weight=0.6, llm_risk_weight=0.4):
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
    # Preserve the planned directed graph contract even before semantic-memory
    # and LLM priors are available. Each current new type gets one edge to each
    # old type, carrying the target class's observed training evidence.
    for source in state.new_types:
        for target in state.old_types:
            node = risk_map.nodes.get(target, RiskNode(target))
            prior_risk, prior_types, prior_reason = (semantic_memory.rule_risk(source, target)
                                                     if semantic_memory is not None else (0.0, [], ""))
            llm_edge = (qwen_assessment or {}).get((source, target), {})
            components = [llm_edge.get(name, 0) for name in
                          ("semantic_overlap", "annotation_conflict", "context_overlap")]
            llm_risk = 0.05 + 0.95 * (0.30 * components[0] + 0.45 * components[1]
                                      + 0.25 * components[2]) / 2.0
            has_llm = bool(llm_edge)
            initial_risk = ((rule_risk_weight * prior_risk + llm_risk_weight * llm_risk)
                            / (rule_risk_weight + llm_risk_weight)) if has_llm else prior_risk
            fused_risk = max(node.score, initial_risk)
            risk_map.edges.append(RiskEdge(
                source=source, target=target, risk=fused_risk,
                risk_type=list(dict.fromkeys(prior_types + llm_edge.get("reason_tags", [])
                                              + list(node.reasons))),
                evidence=dict(node.evidence, rule_risk=prior_risk, llm_risk=llm_risk,
                              initial_risk=initial_risk),
                reason=prior_reason or ("Observed evidence for old target %s; source-specific "
                                        "semantic prior is not configured." % target),
                source_kind="reviewed_rule_llm_plus_observed_evidence" if has_llm
                else "reviewed_rule_plus_observed_evidence" if semantic_memory
                else "observed_training_evidence"))
    return risk_map
