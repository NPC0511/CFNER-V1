"""Frozen task-level risk policy and per-token logits-KD weighting."""

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List

ENTITY_PREFIXES = ("B-", "I-", "E-", "S-")


@dataclass
class RiskKDPolicy:
    task_id: int
    target_risks: Dict[str, float] = field(default_factory=dict)
    label_weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def build_risk_kd_policy(task_id, risk_edges, label_list, old_types,
                         medium_threshold=0.50, high_threshold=0.75,
                         medium_weight=1.15, high_weight=1.30):
    """Freeze target old-class KD weights from directed edge final risks."""
    target_risks = {name: 0.0 for name in old_types}
    for edge in risk_edges:
        target = edge.target if hasattr(edge, "target") else edge["target"]
        risk = edge.final_risk if hasattr(edge, "final_risk") else edge["final_risk"]
        if target in target_risks:
            target_risks[target] = max(target_risks[target], float(risk))
    label_weights = {}
    for label in label_list:
        entity_type = label[2:] if label[:2] in ENTITY_PREFIXES else ""
        risk = target_risks.get(entity_type)
        if risk is None:
            continue
        if risk >= high_threshold:
            label_weights[label] = float(high_weight)
        elif risk >= medium_threshold:
            label_weights[label] = float(medium_weight)
        else:
            label_weights[label] = 1.0
    return RiskKDPolicy(int(task_id), target_risks, label_weights)


def label_id_weights(policy, label_list):
    """Convert a serializable policy to teacher-label-index weights."""
    return {index: float(policy.label_weights.get(label, 1.0))
            for index, label in enumerate(label_list)}


def weighted_kl_by_teacher_label(student_log_scores, teacher_probabilities,
                                 teacher_label_ids, label_weights):
    """Return token-mean baseline and risk-weighted KL terms plus diagnostics."""
    import torch.nn.functional as F
    per_token = F.kl_div(student_log_scores, teacher_probabilities,
                         reduction="none").sum(dim=-1)
    weights = torch.ones_like(per_token)
    for label_id, value in label_weights.items():
        weights = torch.where(teacher_label_ids == int(label_id),
                              weights.new_full((), float(value)), weights)
    return per_token.mean(), (per_token * weights).mean(), weights
