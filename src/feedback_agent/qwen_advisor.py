"""Task-level Qwen semantic-risk advisor with validated cached output."""

import json
import os


class QwenRiskAdvisor(object):
    """Generate semantic components; Python owns risk arithmetic and actions."""

    COMPONENTS = ("semantic_overlap", "annotation_conflict", "context_overlap")
    PLACEHOLDER_TAGS = {"tag", "short_reason_tag"}

    def __init__(self, output_dir, enabled=False, model_path=None,
                 temperature=0.0, max_new_tokens=512, dtype="float32"):
        self.output_dir = output_dir
        self.enabled = bool(enabled)
        self.model_path = model_path
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.dtype = dtype
        self.model = None
        self.tokenizer = None

    def analyze_task(self, task_id, domain, new_types, old_types, memory):
        cache_dir = os.path.join(self.output_dir, "semantic_cache", "qwen")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "%s_task_%d.json" % (domain, task_id))
        result = {"task_id": int(task_id), "domain": domain, "enabled": self.enabled,
                  "status": "disabled", "edges": [], "edge_results": []}
        if not self.enabled:
            return self._write(cache_path, result)
        if not self.model_path or not os.path.isdir(self.model_path):
            result.update(status="fallback", error="qwen_model_path_unavailable")
            return self._write(cache_path, result)
        try:
            self._load_model()
            for source in new_types:
                for target in old_types:
                    edge_result = self._analyze_edge(
                        cache_dir, domain, task_id, source, target, memory)
                    result["edge_results"].append(edge_result)
                    if edge_result.get("status") == "ok":
                        result["edges"].append(edge_result["edge"])
            result["status"] = ("ok" if len(result["edges"]) == len(new_types) * len(old_types)
                                else "partial" if result["edges"] else "fallback")
        except Exception as exc:
            result.update(status="fallback", error="%s: %s" %
                          (type(exc).__name__, str(exc)))
        return self._write(cache_path, result)

    def _analyze_edge(self, cache_dir, domain, task_id, source, target, memory):
        filename = "%s_task_%d_%s_to_%s.json" % (domain, task_id, source, target)
        cache_path = os.path.join(cache_dir, filename)
        if os.path.isfile(cache_path):
            with open(cache_path, encoding="utf-8") as handle:
                cached = json.load(handle)
            if cached.get("status") == "ok" and not self._is_template_edge(
                    cached.get("edge") or {}):
                return cached
        result = {"source": source, "target": target, "status": "fallback",
                  "edge": None, "raw_output": ""}
        try:
            raw_output = self._generate(self._edge_prompt(source, target, memory))
            edge = self._parse_edge(raw_output, source, target)
            result.update(status="ok", edge=edge, raw_output=raw_output)
        except Exception as exc:
            result.update(error="%s: %s" % (type(exc).__name__, str(exc)),
                          raw_output=locals().get("raw_output", ""))
        return self._write(cache_path, result)

    def _load_model(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype = torch.float16 if self.dtype == "float16" and torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model
        self.model.eval()

    def _edge_prompt(self, source, target, memory):
        definitions = memory.entity_types.get("entity_types", {}) if memory is not None else {}
        rules = memory.annotation_rules.get("rules", []) if memory is not None else []
        relevant_rules = [rule for rule in rules
                          if rule.get("source") == source and rule.get("target") == target]
        schema = {"source": source, "target": target,
            "semantic_overlap": "integer 0, 1, or 2",
            "annotation_conflict": "integer 0, 1, or 2",
            "context_overlap": "integer 0, 1, or 2",
            "reason_tags": ["specific non-placeholder reason tag"]}
        return (
            "You analyze one directed interference-risk edge for continual NER. Return one JSON object only, with no markdown. "
            "Use exactly source=%s and target=%s. Assign integer components 0, 1, or 2. "
            "Do not suggest loss weights or training actions.\nDefinitions: %s\nReviewed rules: %s\n"
            "Required schema: %s" % (source, target, json.dumps(definitions),
                                      json.dumps(relevant_rules), json.dumps(schema)))

    def _generate(self, prompt):
        import torch
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=True)
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {name: value.to(self.model.device) for name, value in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(**inputs, do_sample=False,
                                         max_new_tokens=self.max_new_tokens)
        generated = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _parse_edge(self, raw_output, source, target):
        start, end = raw_output.find("{"), raw_output.rfind("}")
        if start < 0 or end < start:
            raise ValueError("missing_json_object")
        payload = json.loads(raw_output[start:end + 1])
        if payload.get("source") != source or payload.get("target") != target:
            raise ValueError("unexpected_edge_identity")
        parsed = {"source": source, "target": target}
        for name in self.COMPONENTS:
            value = payload.get(name)
            if not isinstance(value, int) or value not in (0, 1, 2):
                raise ValueError("invalid_%s" % name)
            parsed[name] = value
        tags = payload.get("reason_tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError("invalid_reason_tags")
        parsed["reason_tags"] = tags
        if self._is_template_edge(parsed):
            raise ValueError("template_rejected")
        return parsed

    def _is_template_edge(self, edge):
        return (all(edge.get(name) == 0 for name in self.COMPONENTS)
                and bool(edge.get("reason_tags"))
                and all(tag in self.PLACEHOLDER_TAGS
                        for tag in edge.get("reason_tags", [])))

    @staticmethod
    def _write(path, result):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=True, indent=2)
        return result
