"""Task-level Qwen semantic-risk advisor with validated cached output."""

import json
import os


class QwenRiskAdvisor(object):
    """Generate semantic components; Python owns risk arithmetic and actions."""

    COMPONENTS = ("semantic_overlap", "annotation_conflict", "context_overlap")

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
        if os.path.isfile(cache_path):
            with open(cache_path, encoding="utf-8") as handle:
                return json.load(handle)
        result = {"task_id": int(task_id), "domain": domain, "enabled": self.enabled,
                  "status": "disabled", "edges": [], "raw_output": ""}
        if not self.enabled:
            return self._write(cache_path, result)
        if not self.model_path or not os.path.isdir(self.model_path):
            result.update(status="fallback", error="qwen_model_path_unavailable")
            return self._write(cache_path, result)
        try:
            self._load_model()
            prompt = self._prompt(new_types, old_types, memory)
            raw_output = self._generate(prompt)
            parsed = self._parse(raw_output, new_types, old_types)
            result.update(status="ok", edges=parsed, raw_output=raw_output)
        except Exception as exc:
            result.update(status="fallback", error="%s: %s" %
                          (type(exc).__name__, str(exc)), raw_output="")
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

    def _prompt(self, new_types, old_types, memory):
        definitions = memory.entity_types.get("entity_types", {}) if memory is not None else {}
        rules = memory.annotation_rules.get("rules", []) if memory is not None else []
        relevant_rules = [rule for rule in rules
                          if rule.get("source") in new_types and rule.get("target") in old_types]
        schema = {
            "risk_edges": [{
                "source": "new_type", "target": "old_type",
                "semantic_overlap": 0, "annotation_conflict": 0,
                "context_overlap": 0, "reason_tags": ["tag"]
            }]
        }
        return (
            "You analyze directed interference risk for continual NER. Return JSON only. "
            "For every requested new-to-old pair, assign integer components 0, 1, or 2. "
            "Do not suggest loss weights or training actions.\n"
            "New types: %s\nOld types: %s\nDefinitions: %s\nReviewed rules: %s\n"
            "Required schema: %s" % (json.dumps(new_types), json.dumps(old_types),
                                      json.dumps(definitions), json.dumps(relevant_rules),
                                      json.dumps(schema)))

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

    def _parse(self, raw_output, new_types, old_types):
        start, end = raw_output.find("{"), raw_output.rfind("}")
        if start < 0 or end < start:
            raise ValueError("missing_json_object")
        payload = json.loads(raw_output[start:end + 1])
        raw_edges = payload.get("risk_edges")
        if not isinstance(raw_edges, list):
            raise ValueError("risk_edges_not_list")
        expected = {(source, target) for source in new_types for target in old_types}
        parsed = {}
        for edge in raw_edges:
            source, target = edge.get("source"), edge.get("target")
            if (source, target) not in expected:
                continue
            components = {}
            for name in self.COMPONENTS:
                value = edge.get(name)
                if not isinstance(value, int) or value not in (0, 1, 2):
                    raise ValueError("invalid_%s" % name)
                components[name] = value
            tags = edge.get("reason_tags", [])
            if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
                raise ValueError("invalid_reason_tags")
            parsed[(source, target)] = dict(components, reason_tags=tags)
        if set(parsed) != expected:
            raise ValueError("missing_expected_edges")
        return [{"source": source, "target": target, **parsed[(source, target)]}
                for source, target in sorted(expected)]

    @staticmethod
    def _write(path, result):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=True, indent=2)
        return result
