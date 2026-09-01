"""Observe-only training monitor for the original INER-RDP loop."""

import json
import csv
import os
import time

from .state import OldClassState, TrainingState
from .semantic_risk import build_risk_map
from .reflection import ReflectionMemory


class FeedbackMonitor(object):
    """Collect small CPU summaries without changing training state."""

    def __init__(self, output_dir, enabled=False, observe_interval_steps=200,
                 feature_probe_max_tokens=500, summary_enabled=True,
                 prototype_stats_enabled=True,
                 structured_state_logging_enabled=True,
                 teacher_confusion_enabled=True,
                 pseudo_uncertainty_enabled=True,
                 prototype_similarity_enabled=True,
                 gradient_conflict_enabled=True, reflection_enabled=True,
                 reflection_forgetting_threshold=1.0):
        self.output_dir = output_dir
        self.enabled = bool(enabled)
        self.summary_enabled = bool(summary_enabled)
        self.prototype_stats_enabled = bool(prototype_stats_enabled)
        self.structured_state_logging_enabled = bool(structured_state_logging_enabled)
        self.teacher_confusion_enabled = bool(teacher_confusion_enabled)
        self.pseudo_uncertainty_enabled = bool(pseudo_uncertainty_enabled)
        self.prototype_similarity_enabled = bool(prototype_similarity_enabled)
        self.gradient_conflict_enabled = bool(gradient_conflict_enabled)
        self.reflection_memory = ReflectionMemory(
            output_dir, enabled=reflection_enabled,
            forgetting_threshold=reflection_forgetting_threshold)
        self.observe_interval_steps = max(int(observe_interval_steps), 1)
        self.task_summary = None
        self.task_path = ""
        self.feature_probe_max_tokens = max(int(feature_probe_max_tokens), 1)
        self.feature_probe = []
        self.feature_reference_means = {}
        self.summary_csv_path = ""
        self.entity_types = []
        self.state = None
        self.action_records = []

    def begin_task(self, task_id, domain, new_types, old_types, label_list):
        if not self.enabled and not self.summary_enabled:
            return
        self.task_path = ""
        if self.enabled:
            task_dir = os.path.join(self.output_dir, "feedback_monitor")
            os.makedirs(task_dir, exist_ok=True)
            self.task_path = os.path.join(task_dir, "%s_task_%d.jsonl" % (domain, int(task_id)))
        self.label_list = list(label_list)
        self.old_types = list(old_types)
        self.new_types = list(new_types)
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
        self.state = TrainingState(
            task_id=int(task_id), domain=str(domain), new_types=list(new_types),
            old_types=list(old_types), seen_types=list(old_types) + list(new_types),
            old_classes={name: OldClassState(name) for name in old_types})
        self.action_records = []
        if self.task_path:
            with open(self.task_path, "w", encoding="utf-8"):
                pass

    def record_action(self, action_record):
        """Append a proposed/validated action without applying it to training."""
        if (not self.enabled or not self.prototype_stats_enabled
                or self.task_summary is None or not self.task_path):
            return
        record = action_record.to_dict()
        record["timestamp"] = time.time()
        record["record_type"] = "action"
        with open(self.task_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        self.task_summary["action_records"] = self.task_summary.get("action_records", 0) + 1
        self.action_records.append(record)

    def record_prototype_stats(self, prototypes, variances, counts):
        """Persist prototype mean/variance/count without affecting training."""
        if (not self.enabled or not self.teacher_confusion_enabled
                or self.task_summary is None or not self.task_path):
            return
        stats = {}
        for index, count in enumerate(counts.detach().cpu().tolist()):
            if count <= 0:
                continue
            stats[str(index)] = {
                "count": int(count),
                "mean_l2": float(prototypes[index].detach().norm().cpu().item()),
                "variance_mean": float(variances[index].detach().mean().cpu().item()),
                "variance_l2": float(variances[index].detach().norm().cpu().item()),
            }
        record = {"timestamp": time.time(), "record_type": "prototype_stats",
                  "prototype_stats": stats}
        with open(self.task_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        self.task_summary["prototype_classes_observed"] = len(stats)
        self.task_summary["prototype_low_count_classes"] = sum(1 for item in stats.values() if item["count"] < 30)

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
        if self.state is not None:
            self.state.step = int(step)
            self.state.metrics["feature_drift"] = dict(drift)
            for entity_type, value in drift.items():
                if entity_type in self.state.old_classes:
                    self.state.old_classes[entity_type].feature_drift = float(value)
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
        if self.state is not None:
            self.state.step = int(step)
            self.state.observation_index += 1
            self.state.metrics.update({
                "mean_confidence": record["mean_confidence"],
                "total_loss": record["total_loss"],
                "valid_tokens": record["valid_tokens"]})
        summary["observations"] += 1
        summary["valid_tokens"] += record["valid_tokens"]
        summary["non_o_tokens"] += record["non_o_tokens"]
        summary["confidence_sum"] += record["mean_confidence"] * record["valid_tokens"]
        if record["total_loss"] is not None:
            summary["loss_sum"] += record["total_loss"]
            summary["loss_count"] += 1
        for name, count in counts.items():
            summary["label_counts"][name] = summary["label_counts"].get(name, 0) + count

    def observe_teacher_confusion(self, step, labels, teacher_logits,
                                  old_types, new_types):
        """Measure teacher predictions of old classes on gold new-class tokens."""
        if not self.enabled or self.task_summary is None or not self.task_path:
            return None
        import torch
        labels_cpu = labels.detach().view(-1).cpu()
        logits_cpu = teacher_logits.detach().view(-1, teacher_logits.shape[-1]).cpu()
        predictions = logits_cpu.argmax(dim=-1)
        old_ids = {i for i, name in enumerate(self.label_list)
                   if name[2:] in old_types and name[:2] in ("B-", "I-", "E-", "S-")}
        result = {}
        for entity_type in new_types:
            new_ids = {i for i, name in enumerate(self.label_list)
                       if name[2:] == entity_type and name[:2] in ("B-", "I-", "E-", "S-")}
            mask = torch.zeros_like(labels_cpu, dtype=torch.bool)
            for label_id in new_ids:
                mask |= labels_cpu == label_id
            count = int(mask.sum().item())
            confusion = (float(sum(1 for value in predictions[mask].tolist()
                                   if value in old_ids)) / count) if count else None
            result[entity_type] = confusion
        record = {"timestamp": time.time(), "step": int(step),
                  "teacher_confusion": result,
                  "teacher_confusion_counts": {
                      key: int(sum(1 for i in labels_cpu.tolist()
                                   if i in {j for j, name in enumerate(self.label_list)
                                            if name[2:] == key and name[:2] in ("B-", "I-", "E-", "S-")}))
                      for key in new_types}}
        with open(self.task_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        self.task_summary["latest_teacher_confusion"] = result
        if self.state is not None:
            self.state.step = int(step)
            self.state.metrics["teacher_confusion"] = dict(result)
            for entity_type, value in result.items():
                if entity_type in self.state.old_classes:
                    self.state.old_classes[entity_type].teacher_confusion = value
        return result

    def observe_pseudo_uncertainty(self, step, labels, pseudo_labels,
                                   pseudo_confidence, pseudo_probabilities):
        """Record confidence, entropy and coverage for old-class pseudo labels."""
        if (not self.enabled or not self.pseudo_uncertainty_enabled
                or self.task_summary is None or not self.task_path):
            return None
        import torch
        labels_cpu = labels.detach().view(-1).cpu()
        pseudo_cpu = pseudo_labels.detach().view(-1).cpu()
        confidence_cpu = pseudo_confidence.detach().view(-1).cpu()
        probs_cpu = pseudo_probabilities.detach().view(-1, pseudo_probabilities.shape[-1]).cpu()
        entropy = -(probs_cpu.clamp_min(1e-12) * probs_cpu.clamp_min(1e-12).log()).sum(dim=-1)
        old_mask = (labels_cpu == 0) & (labels_cpu != -100)
        result = {"mean_confidence": float(confidence_cpu[old_mask].mean().item()) if torch.any(old_mask) else None,
                  "mean_entropy": float(entropy[old_mask].mean().item()) if torch.any(old_mask) else None,
                  "coverage": {}}
        candidate_count = int(old_mask.sum().item())
        for index, name in enumerate(self.label_list):
            if index == 0 or not name or name[:2] not in ("B-", "I-", "E-", "S-"):
                continue
            count = int(((pseudo_cpu == index) & old_mask).sum().item())
            result["coverage"][name] = {"count": count,
                                         "ratio": (float(count) / candidate_count)
                                         if candidate_count else None}
        record = {"timestamp": time.time(), "step": int(step), "pseudo_uncertainty": result}
        with open(self.task_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        self.task_summary["latest_pseudo_uncertainty"] = result
        if self.state is not None:
            self.state.step = int(step)
            self.state.metrics["pseudo_uncertainty"] = result
        return result

    def observe_prototype_similarity(self, step, feature_model, prototypes):
        """Compare current probe feature means with stored old-class prototypes."""
        if (not self.enabled or not self.prototype_similarity_enabled
                or self.task_summary is None or not self.task_path
                or not self.feature_probe):
            return None
        import torch
        device = next(feature_model.parameters()).device
        sums, counts = {}, {}
        was_training = feature_model.training
        feature_model.eval()
        with torch.no_grad():
            for inputs_cpu, labels_cpu in self.feature_probe:
                features = feature_model.forward_encoder(inputs_cpu.to(device)).detach().cpu()
                labels_cpu = labels_cpu.view(-1)
                features = features.view(-1, features.shape[-1])
                for entity_type in self.old_types:
                    ids = [i for i, name in enumerate(self.label_list)
                           if name[2:] == entity_type and name[:2] in ("B-", "I-", "E-", "S-")]
                    mask = torch.zeros_like(labels_cpu, dtype=torch.bool)
                    for label_id in ids:
                        mask |= labels_cpu == label_id
                    if torch.any(mask):
                        sums[entity_type] = sums.get(entity_type, torch.zeros(features.shape[-1])) + features[mask].sum(0)
                        counts[entity_type] = counts.get(entity_type, 0) + int(mask.sum().item())
        feature_model.train(was_training)
        prototype_cpu = prototypes.detach().cpu()
        result = {}
        for entity_type, feature_sum in sums.items():
            ids = [i for i, name in enumerate(self.label_list)
                   if name[2:] == entity_type and name[:2] in ("B-", "I-", "E-", "S-") and i < prototype_cpu.shape[0]]
            if not ids:
                continue
            current = feature_sum / counts[entity_type]
            target = prototype_cpu[ids].mean(0)
            result[entity_type] = float(torch.nn.functional.cosine_similarity(current[None], target[None]).item())
        record = {"timestamp": time.time(), "step": int(step),
                  "prototype_similarity": result, "prototype_similarity_counts": counts}
        with open(self.task_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        self.task_summary["latest_prototype_similarity"] = result
        if self.state is not None:
            self.state.step = int(step)
            self.state.metrics["prototype_similarity"] = dict(result)
            for entity_type, value in result.items():
                if entity_type in self.state.old_classes:
                    self.state.old_classes[entity_type].prototype_similarity = value
        return result

    def observe_gradient_conflict(self, step, new_loss, old_loss, parameters):
        """Measure conflict between new CE and old distillation gradients."""
        if (not self.enabled or not self.gradient_conflict_enabled
                or self.task_summary is None or not self.task_path
                or int(step) % self.observe_interval_steps
                or new_loss is None or old_loss is None):
            return None
        import torch
        params = [p for p in parameters if p.requires_grad]
        if not params or not new_loss.requires_grad or not old_loss.requires_grad:
            return None
        new_grads = torch.autograd.grad(new_loss, params, retain_graph=True,
                                        allow_unused=True)
        old_grads = torch.autograd.grad(old_loss, params, retain_graph=True,
                                        allow_unused=True)
        new_vec = torch.cat([g.detach().reshape(-1).cpu() for g in new_grads if g is not None])
        old_vec = torch.cat([g.detach().reshape(-1).cpu() for g in old_grads if g is not None])
        if not new_vec.numel() or not old_vec.numel():
            return None
        cosine = torch.nn.functional.cosine_similarity(new_vec[None], old_vec[None]).item()
        conflict = max(0.0, -float(cosine))
        result = {"cosine": float(cosine), "conflict": conflict}
        record = {"timestamp": time.time(), "step": int(step),
                  "gradient_conflict": result}
        with open(self.task_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        self.task_summary["latest_gradient_conflict"] = result
        if self.state is not None:
            self.state.step = int(step)
            self.state.metrics["gradient_conflict"] = result
        return result

    def end_task(self, metrics=None):
        if self.task_summary is None:
            return ""
        summary = dict(self.task_summary)
        summary["metrics"] = dict(metrics or {})
        summary["mean_confidence"] = summary["confidence_sum"] / float(summary["valid_tokens"]) if summary["valid_tokens"] else None
        summary["mean_loss"] = summary["loss_sum"] / float(summary["loss_count"]) if summary["loss_count"] else None
        if self.state is not None:
            per_class_f1 = dict((metrics or {}).get("per_class_f1", {}))
            f1_before = dict((metrics or {}).get("f1_before", {}))
            for entity_type, item in self.state.old_classes.items():
                if entity_type in per_class_f1:
                    item.f1_current = float(per_class_f1[entity_type])
                if entity_type in f1_before:
                    item.f1_before = float(f1_before[entity_type])
                    if item.f1_current is not None:
                        item.f1_drop = item.f1_before - item.f1_current
        if self.state is not None and self.structured_state_logging_enabled:
            summary["state"] = self.state.to_dict()
        for key in ("confidence_sum", "loss_sum", "loss_count"):
            summary.pop(key, None)
        path = self.task_path[:-6] + "summary.json"
        if self.task_path:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=True, indent=2)
        if self.summary_enabled and summary.get("metrics", {}).get("status") == "completed":
            self._append_summary_csv(summary["metrics"])
        reflection_path = self.reflection_memory.write(
            self.state, summary.get("latest_semantic_risk"), self.action_records)
        if reflection_path:
            summary["reflection_path"] = reflection_path
            if self.task_path:
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(summary, handle, ensure_ascii=True, indent=2)
        self.task_summary = None
        return path if self.task_path else self.summary_csv_path

    def get_state(self):
        """Return the latest structured snapshot for controller consumers."""
        return self.state

    def record_semantic_risk(self, drift_threshold=0.15, confusion_threshold=0.20,
                             entropy_threshold=1.0, similarity_threshold=0.50):
        """Persist the current rule-based risk map without taking an action."""
        if (not self.enabled or not self.task_path or self.state is None
                or not self.task_summary):
            return None
        risk_map = build_risk_map(self.state, drift_threshold, confusion_threshold,
                                  entropy_threshold, similarity_threshold)
        with open(self.task_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": time.time(),
                                     "record_type": "semantic_risk",
                                     "semantic_risk": risk_map.to_dict()},
                                    ensure_ascii=True) + "\n")
        self.task_summary["latest_semantic_risk"] = risk_map.to_dict()
        return risk_map

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
