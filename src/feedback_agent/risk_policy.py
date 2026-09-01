"""Task-start graph construction and frozen risk-weighted KD policy."""

from .risk_kd import build_risk_kd_policy
from .semantic_risk import build_task_risk_graph


def build_task_policy(task_id, domain, new_types, old_types, label_list,
                      semantic_memory, qwen_assessment, rule_risk_weight,
                      llm_risk_weight, medium_threshold, high_threshold,
                      medium_weight, high_weight):
    graph = build_task_risk_graph(
        task_id=task_id, domain=domain, new_types=new_types, old_types=old_types,
        semantic_memory=semantic_memory, qwen_assessment=qwen_assessment,
        rule_risk_weight=rule_risk_weight, llm_risk_weight=llm_risk_weight)
    policy = build_risk_kd_policy(
        task_id, graph.edges, label_list, old_types, medium_threshold,
        high_threshold, medium_weight, high_weight)
    graph.training_policy = {"old_label_weights": policy.label_weights,
                             "target_risks": policy.target_risks}
    return graph, policy
