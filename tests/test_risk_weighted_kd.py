import unittest

try:
    import torch
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

from src.feedback_agent.qwen_advisor import QwenRiskAdvisor
from src.feedback_agent.risk_kd import build_risk_kd_policy, build_prototype_anchor_targets
from src.feedback_agent.semantic_risk import RiskEdge


class RiskPolicyTests(unittest.TestCase):
    def test_high_risk_target_only_raises_its_bio_labels(self):
        edges = [RiskEdge("organisation", "location", final_risk=0.85)]
        policy = build_risk_kd_policy(
            2, edges, ["O", "B-location", "I-location", "B-misc", "I-misc"],
            ["location", "misc"])
        self.assertEqual(policy.label_weights["B-location"], 1.30)
        self.assertEqual(policy.label_weights["I-location"], 1.30)
        self.assertEqual(policy.label_weights["B-misc"], 1.00)
        self.assertEqual(policy.label_weights["I-misc"], 1.00)

    def test_template_qwen_output_is_rejected(self):
        advisor = QwenRiskAdvisor("unused")
        raw = ('{"source":"misc","target":"location","semantic_overlap":0,'
               '"annotation_conflict":0,"context_overlap":0,'
               '"reason_tags":["short_reason_tag"]}')
        with self.assertRaisesRegex(ValueError, "template_rejected"):
            advisor._parse_edge(raw, "misc", "location")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for prototype tests")
    def test_anchor_selects_only_observed_finite_old_labels(self):
        prototypes = torch.tensor([[0., 0.], [1., 1.], [float("nan"), 1.]])
        counts = torch.tensor([0, 3, 4])
        selected = build_prototype_anchor_targets(
            prototypes, counts, {1: 1.30, 2: 1.15})
        self.assertEqual(selected, [(1, 1.30, 3)])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for KD tests")
class RiskWeightedKDNumericalTests(unittest.TestCase):
    def test_weighted_kl_only_increases_high_risk_teacher_label(self):
        from src.feedback_agent.risk_kd import weighted_kl_by_teacher_label
        student = torch.log(torch.tensor([[0.6, 0.4], [0.6, 0.4]]))
        teacher = torch.tensor([[0.2, 0.8], [0.2, 0.8]])
        labels = torch.tensor([0, 1])
        baseline, weighted, weights = weighted_kl_by_teacher_label(
            student, teacher, labels, {1: 1.30})
        self.assertGreater(weighted.item(), baseline.item())
        self.assertEqual(weights.tolist(), [1.0, 1.30])


if __name__ == "__main__":
    unittest.main()
