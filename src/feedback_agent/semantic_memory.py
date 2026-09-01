"""Read-only loader for reviewed semantic priors."""

import json
import os


class SemanticMemory(object):
    def __init__(self, root_dir, dataset):
        self.directory = os.path.join(root_dir, dataset)
        with open(os.path.join(self.directory, "entity_types.json"), encoding="utf-8") as handle:
            self.entity_types = json.load(handle)
        with open(os.path.join(self.directory, "annotation_rules.json"), encoding="utf-8") as handle:
            self.annotation_rules = json.load(handle)

    def rules_for(self, source, target):
        return [rule for rule in self.annotation_rules.get("rules", [])
                if rule.get("source") == source and rule.get("target") == target]

    def rule_risk(self, source, target):
        rules = self.rules_for(source, target)
        if not rules:
            return 0.0, [], ""
        rule = max(rules, key=lambda item: float(item.get("weight", 0.0)))
        # Reviewed rule weights are intentionally mapped to a bounded prior.
        risk = min(max(float(rule.get("weight", 0.0)) / 2.0, 0.0), 1.0)
        return risk, list(rule.get("risk_type", [])), rule.get("reason", "")
