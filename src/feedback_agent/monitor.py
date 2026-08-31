"""Observe-only training monitor for the original INER-RDP loop."""

import json
import csv
import os
import time


class FeedbackMonitor(object):
    """Collect small CPU summaries without changing training state."""

    def __init__(self, output_dir, enabled=False, observe_interval_steps=200,
                 feature_probe_max_tokens=500, summary_enabled=True):
        self.output_dir = output_dir
        self.enabled = bool(enabled)
        self.summary_enabled = bool(summary_enabled)
        self.observe_interval_steps = max(int(observe_interval_steps), 1)
        self.task_summary = None
        self.task_path = ""
        self.feature_probe_max_tokens = max(int(feature_probe_max_tokens), 1)
        self.feature_probe = []
        self.feature_reference_means = {}
        self.summary_csv_path = ""
        self.entity_types = []

    def begin_task(self, task_id, domain, new_types, old_types, label_list):
        if not self.enabled and not self.summary_enabled:
            return
        self.task_path = ""
        if self.enabled:
            task_dir = os.path.join(self.output_dir, "feedback_monitor")
            os.makedirs(task_dir, exist_ok=True)
            self.task_path = os.path.join(task_dir, "%s_task_%d.jsonl" % (domain, int(task_id)))
        self.label_list = list(label_list)
        self.entity_types = sorted({name[2:] for name in self.label_list
                                    if len(name) > 2 and name[:2] in ("B-", "I-", "E-", "S-")})
        self.summary_csv_path = os.path.join(self.output_dir, "summary.csv")
        self.task_summary = {
            "task_id": int(task_id), "domain": str(domain),
            "new_types": list(new_types), "old_types": list(old_types),
            "observations": 0, "valid_tokens": 0, "non_o_tokens": 0,
            "label_counts": {}, "confidence_sum": 0.0,
            "loss_sum": 0.0, "loss_count": 0,
            "feature_probe_tokens": 0, "feature_drift_observations": 0
        }
        if self.task_path:
            with open(self.task_path, "w", encoding="utf-8"):
                pass

    def record_action(self, action_record):
        """Append a proposed/validated action without applying it to training."""
        if not self.enabled or self.task_summary is None or not self.task_path:
            return
        record = action_record.to_dict()
        record["timestamp"] = time.time()
        record["record_type"] = "action"
        with open(self.task_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        self.task_summary["action_records"] = self.task_summary.get("action_records", 0) + 1

    def capture_feature_probe(self, dataloader, reference_model, old_types):
        """Freeze an old-class probe and its teacher feature means for P2."""
        if not self.enabled or self.task_summary is None or not old_types:
            return
        import torch

        remaining = self.feature_probe_max_tokens
        sums = {}
        counts = {}
        was_training = reference_model.training
        reference_model.eval()
        with torch.no_grad():
            for inputs, labels in dataloader:
                if remaining <= 0:
                    break
                inputs = inputs.cuda()
                labels = labels.cuda()
                features = reference_model.forward_encoder(inputs)
                selected = torch.zeros_like(labels, dtype=torch.bool)
                for entity_type in old_types:
                    label_ids = [i for i, name in enumerate(self.label_list)
                                 if name[2:] == entity_type and name[:2] in ("B-", "I-", "E-", "S-")]
                    if not label_ids:
                        continue
                    mask = torch.zeros_like(labels, dtype=torch.bool)
                    for label_id in label_ids:
                        mask |= labels == label_id
                    if torch.any(mask):
                        values = features[mask].detach().cpu()
                        sums[entity_type] = sums.get(entity_type, torch.zeros(values.shape[-1])) + values.sum(dim=0)
                        counts[entity_type] = counts.get(entity_type, 0) + int(values.shape[0])
                        selected |= mask
                token_count = int(selected.sum().item())
                if not token_count:
                    continue
                self.feature_probe.append((inputs.detach().cpu(), labels.detach().cpu()))
                remaining -= token_count
        reference_model.train(was_training)
        self.feature_reference_means = {
            entity_type: (sums[entity_type] / counts[entity_type]).tolist()
            for entity_type in sums if counts[entity_type]
        }
        self.task_summary["feature_probe_tokens"] = sum(counts.values())
        self.task_summary["feature_probe_counts"] = counts

    def observe_feature_drift(self, step, feature_model):
        """Record 1-cosine teacher/current means on the frozen P2 probe."""
        if (not self.enabled or self.task_summary is None or not self.feature_probe
                or int(step) % self.observe_interval_steps):
            return
        import torch
        device = next(feature_model.parameters()).device
        sums, counts = {}, {}
        was_training = feature_model.training
        feature_model.eval()
        with torch.no_grad():
            for inputs_cpu, labels_cpu in self.feature_probe:
                inputs, labels = inputs_cpu.to(device), labels_cpu.to(device)
                features = feature_model.forward_encoder(inputs)
                for entity_type, reference_mean in self.feature_reference_means.items():
                    label_ids = [i for i, name in enumerate(self.label_list)
                                 if name[2:] == entity_type and name[:2] in ("B-", "I-", "E-", "S-")]
                    mask = torch.zeros_like(labels, dtype=torch.bool)
                    for label_id in label_ids:
                        mask |= labels == label_id
                    if torch.any(mask):
                        values = features[mask].detach().cpu()
                        sums[entity_type] = sums.get(entity_type, torch.zeros(values.shape[-1])) + values.sum(dim=0)
                        counts[entity_type] = counts.get(entity_type, 0) + int(values.shape[0])
        feature_model.train(was_training)
        drift = {}
        for entity_type, feature_sum in sums.items():
            current_mean = feature_sum / counts[entity_type]
            reference_mean = torch.tensor(self.feature_reference_means[entity_type])
            cosine = torch.nn.functional.cosine_similarity(
                current_mean.unsqueeze(0), reference_mean.unsqueeze(0)).item()
            drift[entity_type] = 1.0 - cosine
        record = {
            "timestamp": time.time(), "step": int(step),
            "feature_drift": drift, "feature_drift_counts": counts
        }
        with open(self.task_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        self.task_summary["feature_drift_observations"] += 1
        self.task_summary["latest_feature_drift"] = drift
        return drift

    def observe_batch(self, step, labels, logits, total_loss=None):
        if not self.enabled or self.task_summary is None or int(step) % self.observe_interval_steps:
            return
        import torch
        with torch.no_grad():
            labels_cpu = labels.detach().view(-1).cpu()
            logits_cpu = logits.detach().view(-1, logits.shape[-1]).cpu()
            valid = labels_cpu != -100
            if not torch.any(valid):
                return
            valid_labels = labels_cpu[valid]
            probabilities = torch.softmax(logits_cpu[valid], dim=-1)
            predictions = probabilities.argmax(dim=-1)
            confidence = probabilities.max(dim=-1).values
            counts = {}
            for label_id in valid_labels.tolist():
                name = self.label_list[int(label_id)] if 0 <= int(label_id) < len(self.label_list) else "<unknown>"
                counts[name] = counts.get(name, 0) + 1
            record = {
                "timestamp": time.time(), "step": int(step),
                "valid_tokens": int(valid_labels.numel()),
                "non_o_tokens": int((valid_labels != 0).sum().item()),
                "label_counts": counts,
                "mean_confidence": float(confidence.mean().item()),
                "non_o_accuracy": None if not torch.any(valid_labels != 0) else float((predictions[valid_labels != 0] == valid_labels[valid_labels != 0]).float().mean().item()),
                "total_loss": None if total_loss is None else float(total_loss)
            }
        with open(self.task_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        summary = self.task_summary
        summary["observations"] += 1
        summary["valid_tokens"] += record["valid_tokens"]
        summary["non_o_tokens"] += record["non_o_tokens"]
        summary["confidence_sum"] += record["mean_confidence"] * record["valid_tokens"]
        if record["total_loss"] is not None:
            summary["loss_sum"] += record["total_loss"]
            summary["loss_count"] += 1
        for name, count in counts.items():
            summary["label_counts"][name] = summary["label_counts"].get(name, 0) + count

    def end_task(self, metrics=None):
        if self.task_summary is None:
            return ""
        summary = dict(self.task_summary)
        summary["metrics"] = dict(metrics or {})
        summary["mean_confidence"] = summary["confidence_sum"] / float(summary["valid_tokens"]) if summary["valid_tokens"] else None
        summary["mean_loss"] = summary["loss_sum"] / float(summary["loss_count"]) if summary["loss_count"] else None
        for key in ("confidence_sum", "loss_sum", "loss_count"):
            summary.pop(key, None)
        path = self.task_path[:-6] + "summary.json"
        if self.task_path:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=True, indent=2)
        if self.summary_enabled and summary.get("metrics", {}).get("status") == "completed":
            self._append_summary_csv(summary["metrics"])
        self.task_summary = None
        return path if self.task_path else self.summary_csv_path

    def _append_summary_csv(self, metrics):
        """Write a human-readable cumulative experiment table."""
        per_class = dict(metrics.get("per_class_f1", {}))
        row = {
            "stage": int(metrics.get("task_id", self.task_summary["task_id"])) + 1,
            "new_entity_types": ";".join(metrics.get("new_types", self.task_summary["new_types"])),
            "seen_entity_types": ";".join(metrics.get("seen_types", [])),
            "micro_f1": metrics.get("test_micro_f1"),
            "macro_f1": metrics.get("test_macro_f1"),
            "seen_class_avg_f1": metrics.get("seen_class_average_f1"),
            "old_class_avg_f1": metrics.get("old_class_average_f1"),
            "new_class_avg_f1": metrics.get("new_class_average_f1")
        }
        for entity_type in self.entity_types:
            row["f1_%s" % entity_type] = per_class.get(entity_type)

        columns = list(row.keys())
        existing = []
        if os.path.isfile(self.summary_csv_path):
            with open(self.summary_csv_path, "r", encoding="utf-8-sig", newline="") as handle:
                existing = list(csv.DictReader(handle))
                if existing:
                    columns = list(existing[0].keys())
                    for key in row:
                        if key not in columns:
                            columns.append(key)
        # Re-running a task replaces its row instead of duplicating it.
        existing = [item for item in existing
                    if item.get("stage") not in (str(row["stage"]), "avg")]
        existing.append({key: row.get(key, "") for key in columns})
        data_rows = existing
        numeric_columns = ["micro_f1", "macro_f1", "seen_class_avg_f1",
                           "old_class_avg_f1", "new_class_avg_f1"] + [
                               "f1_%s" % entity_type for entity_type in self.entity_types]
        average = {key: "avg" if key == "stage" else "" for key in columns}
        for key in numeric_columns:
            values = []
            for item in data_rows:
                try:
                    values.append(float(item.get(key, "")))
                except (TypeError, ValueError):
                    pass
            if values:
                average[key] = "%.4f" % (sum(values) / len(values))
        with open(self.summary_csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data_rows)
            writer.writerow(average)
