"""Explicit opt-in API transport, exact JSON parsing and content-addressed cache."""
import json
import os
import runpy
from pathlib import Path
from .io import digest, write_json

PROMPT_VERSION = "surimi-v1"


def load_provider(name, configured, local_path="local_llm_config.py"):
    """Allow direct local Python configuration without serializing credentials."""
    provider = dict(configured)
    if local_path and Path(local_path).is_file():
        local = runpy.run_path(str(local_path)).get("PROVIDERS", {}).get(name, {})
        provider.update({k: v for k, v in local.items() if v})
    return provider


def redact_error(exc, provider):
    message = f"{type(exc).__name__}: {exc}"
    key = provider.get("api_key")
    if key:
        message = message.replace(key, "[REDACTED]")
    return message


def make_prompt(x, y, query, candidates, representation, wave=None, control="none", seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    x, y, query = np.array(x, copy=True), np.array(y, copy=True), np.array(query, copy=True)
    if control in ("label_permutation", "pair_shuffle"):
        if control == "label_permutation":
            labels = np.unique(y)
            mapping = dict(zip(labels, rng.permutation(labels)))
            y = np.array([mapping[v] for v in y])
        else:
            y = rng.permutation(y)
    if control == "query_band_shuffle":
        query = query[rng.permutation(len(query))]
    if control == "synchronous_band_shuffle":
        order = rng.permutation(len(query))
        x, query = x[:, order], query[order]
        if wave is not None:
            wave = np.array(wave)[order].tolist()
    examples = [{"spectrum": row.round(6).tolist(), "label": float(label)} for row, label in zip(x, y)]
    if control == "label_only":
        examples = [{"label": float(label)} for label in y]
    if control == "prompt_order":
        examples = [examples[i] for i in rng.permutation(len(examples))]
    payload = {"candidate_labels": list(candidates), "representation": representation,
               "examples": examples, "query": {"spectrum": query.round(6).tolist()}}
    if wave is not None:
        payload["wavelength_nm"] = list(wave)
    if control == "label_only":
        payload["query"] = {}
    return [{"role": "system", "content": "Classify the undeclared starch addition level of a surimi spectrum. "
             "Use the labeled reference spectra. Return only a JSON object with exactly one key: "
             "label. Its value must be one candidate label. Do not update model weights."},
            {"role": "user", "content": json.dumps(payload, separators=(",", ":"))}]


def parse_label(text, candidates):
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"label"}:
        raise ValueError("Expected exactly {label: candidate}")
    label = value["label"]
    if isinstance(label, bool) or not isinstance(label, (str, int, float)):
        raise ValueError("Invalid label type")
    label = float(label)
    if label not in candidates:
        raise ValueError("Label outside candidate set")
    return label


class FrozenAPI:
    def __init__(self, spec, cache, allow_api=False, max_requests=0):
        self.spec, self.cache = spec, Path(cache)
        self.allow_api, self.max_requests, self.calls = allow_api, max_requests, 0

    def predict(self, messages):
        request = {"model": self.spec["model"], "messages": messages,
                   "temperature": 0, "max_tokens": 128}
        key = digest({"request": request, "endpoint": self.spec["base_url"], "prompt_version": PROMPT_VERSION})
        path = self.cache / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")), True
        if not self.allow_api or self.calls >= self.max_requests:
            raise RuntimeError("API disabled or request budget exhausted")
        from openai import OpenAI
        api_key = self.spec.get("api_key") or os.environ.get(self.spec.get("api_key_env", ""))
        if not api_key:
            raise RuntimeError("API key is not configured in local_llm_config.py")
        client = OpenAI(api_key=api_key,
                        base_url=self.spec["base_url"], max_retries=0, timeout=60)
        self.calls += 1
        response = client.chat.completions.create(**request)
        result = response.model_dump()
        write_json(path, result)
        return result, False
