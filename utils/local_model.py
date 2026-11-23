"""
Local LLM model wrapper.
Provides an OpenAI SDK-compatible interface for locally loaded HuggingFace models
with LoRA adapters and 4-bit quantization.

Set LOCAL_LLM_API_BASE (e.g. http://127.0.0.1:8080/v1) to serve instead from an
OpenAI-compatible endpoint such as the local Strata engine; no model is loaded
into this process then. LOCAL_LLM_API_MODEL picks the model id (default: the
first one the server lists) and LOCAL_LLM_API_TIMEOUT the request timeout.
"""

import json
import os
import re
import logging
import urllib.error
import urllib.request
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class _Message:
    """Mimics OpenAI ChatCompletionMessage."""
    def __init__(self, content: str, reasoning_content: str = None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.role = "assistant"


class _Choice:
    """Mimics OpenAI Choice."""
    def __init__(self, content: str, reasoning_content: str = None):
        self.message = _Message(content, reasoning_content)
        self.finish_reason = "stop"
        self.index = 0


class _Usage:
    """Mimics OpenAI Usage."""
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _Response:
    """Mimics OpenAI ChatCompletion response."""
    def __init__(self, content: str, reasoning_content: str = None, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.choices = [_Choice(content, reasoning_content)]
        self.usage = _Usage(prompt_tokens, completion_tokens)


class _ChatCompletions:
    """Mimics OpenAI chat.completions endpoint."""
    def __init__(self, model):
        self._model = model

    def create(self, model: str = None, messages: List[Dict[str, str]] = None,
               temperature: float = 0.7, max_tokens: int = 4096,
               prefill_chunk_size: int = None, **kwargs) -> _Response:
        """
        OpenAI-compatible chat completion call.

        Args:
            model: ignored (uses local model)
            messages: list of {"role": ..., "content": ...}
            temperature: sampling temperature
            max_tokens: max new tokens to generate
            prefill_chunk_size: optional chunk size for memory-efficient prompt prefill

        Returns:
            Response object mimicking OpenAI ChatCompletion
        """
        result = self._model.generate(
            messages=messages or [],
            temperature=temperature,
            max_new_tokens=max_tokens,
            prefill_chunk_size=prefill_chunk_size,
        )
        text = result["text"]
        # Extract thinking tokens if present (e.g., <think>...</think>)
        reasoning_content = None
        think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if think_match:
            reasoning_content = think_match.group(1).strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return _Response(
            content=text,
            reasoning_content=reasoning_content,
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
        )


class _Chat:
    """Mimics OpenAI chat endpoint."""
    def __init__(self, model):
        self.completions = _ChatCompletions(model)


class LocalLLM:
    """
    Local HuggingFace model with LoRA adapter and 4-bit quantization.
    Exposes an OpenAI SDK-compatible interface via .chat.completions.create().

    Usage:
        llm = LocalLLM(
            base_model_path="LLM models/Qwen3.5-9B",
            lora_path="LLM models/lora_finetuned_model_qwen3.5-9b_spectral_analysis",
        )
        response = llm.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=1024,
        )
        print(response.choices[0].message.content)
    """

    def __init__(
        self,
        base_model_path: Optional[str] = None,
        lora_path: Optional[str] = None,
        device: str = "auto",
        max_memory: Optional[dict] = None,
        api_base: Optional[str] = None,
        api_model: Optional[str] = None,
    ):
        """
        Load base model with 4-bit quantization and optional LoRA adapter.

        Args:
            base_model_path: path to base HuggingFace model directory
            lora_path: path to LoRA adapter directory (optional)
            device: "auto", "cuda", "cpu". "cuda" keeps the quantized model on one GPU.
            max_memory: GPU memory allocation dict, e.g. {0: "6GiB", "cpu": "16GiB"}
            api_base: an OpenAI-compatible base URL to use instead of loading a
                model here (default: the LOCAL_LLM_API_BASE environment variable)
            api_model: model id to send to that server (default: LOCAL_LLM_API_MODEL,
                else the first id the server lists)
        """
        api_base = api_base or os.environ.get("LOCAL_LLM_API_BASE")
        self._api_base = api_base.rstrip("/") if api_base else None
        self._api_model = api_model or os.environ.get("LOCAL_LLM_API_MODEL")
        self._api_timeout = int(os.environ.get("LOCAL_LLM_API_TIMEOUT", "1800"))
        self.chat = _Chat(self)
        self._model = None
        self._tokenizer = None
        self._base_model_path = base_model_path
        self._lora_path = lora_path
        self._device = device
        self._max_memory = max_memory
        self._loaded = bool(self._api_base)
        if self._api_base:
            logger.info(f"Using the OpenAI-compatible endpoint {self._api_base}")

    def _ensure_loaded(self):
        """Lazy-load model on first use."""
        if self._loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        # Patch bitsandbytes Params4bit for transformers compatibility
        try:
            import bitsandbytes as bnb
            _orig_new = bnb.nn.Params4bit.__new__
            def _patched_new(cls, *args, **kwargs):
                kwargs.pop('_is_hf_initialized', None)
                return _orig_new(cls, *args, **kwargs)
            bnb.nn.Params4bit.__new__ = _patched_new
        except Exception:
            pass

        logger.info(f"Loading base model from {self._base_model_path}...")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._base_model_path,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if self._device == "cpu":
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float32,
                "device_map": "cpu",
            }
            self._model = AutoModelForCausalLM.from_pretrained(
                self._base_model_path, **model_kwargs
            )
        else:
            # 4-bit quantization config
            force_single_gpu = self._device in {"cuda", "cuda:0", "gpu"}
            if force_single_gpu and not torch.cuda.is_available():
                raise RuntimeError("LocalLLM(device='cuda') requires a CUDA GPU")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                # Disable nested quantization so automatic CPU/GPU dispatch
                # does not need to materialize quantization offsets on meta tensors.
                bnb_4bit_use_double_quant=False,
                llm_int8_enable_fp32_cpu_offload=not force_single_gpu,
            )
            model_kwargs = {
                "trust_remote_code": True,
                "quantization_config": bnb_config,
                "device_map": {"": torch.cuda.current_device()} if force_single_gpu else "auto",
            }
            if self._max_memory:
                model_kwargs["max_memory"] = self._max_memory
            self._model = AutoModelForCausalLM.from_pretrained(
                self._base_model_path, **model_kwargs
            )

        # Apply LoRA adapter if provided
        if self._lora_path and Path(self._lora_path).exists():
            logger.info(f"Applying LoRA adapter from {self._lora_path}...")
            self._model = PeftModel.from_pretrained(
                self._model, self._lora_path
            )

        self._model.eval()
        self._loaded = True
        logger.info("Model loaded successfully")

    # Support OpenAI-style constructor: LocalLLM(api_key=..., base_url=...)
    # so it can be used as a drop-in replacement for OpenAI()
    @classmethod
    def from_config(cls, api_key: str = None, base_url: str = None, **kwargs):
        """
        Factory method matching OpenAI(api_key=..., base_url=...) signature.
        A base_url that is an http(s) address selects the API mode; model paths
        come from kwargs.

        Usage:
            client = LocalLLM.from_config(
                api_key="unused",
                base_url="unused",
                base_model_path="LLM models/Qwen3.5-9B",
                lora_path="LLM models/lora_finetuned_model_qwen3.5-9b_spectral_analysis",
            )
        """
        if base_url and str(base_url).startswith(("http://", "https://")):
            kwargs.setdefault("api_base", base_url)
        return cls(**kwargs)

    def _api_model_id(self) -> str:
        """The model id to send: the configured one, else the server's first."""
        if self._api_model:
            return self._api_model
        with urllib.request.urlopen(self._api_base + "/models", timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
        self._api_model = data["data"][0]["id"]
        return self._api_model

    def _generate_api(self, messages, temperature: float, max_new_tokens: int) -> dict:
        """One chat completion from the OpenAI-compatible endpoint."""
        payload = {
            "model": self._api_model_id(),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            self._api_base + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._api_timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"{self._api_base}: HTTP {e.code}: {body}") from e
        message = (data.get("choices") or [{}])[0].get("message") or {}
        usage = data.get("usage") or {}
        return {
            "text": message.get("content") or "",
            "reasoning_content": message.get("reasoning_content"),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
        }

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_new_tokens: int = 4096,
        prefill_chunk_size: int = None,
    ) -> dict:
        """
        Generate a response from the model.

        Args:
            messages: list of {"role": "system"/"user"/"assistant", "content": ...}
            temperature: sampling temperature
            max_new_tokens: max tokens to generate
            prefill_chunk_size: optional chunk size for prompt prefill

        Returns:
            Dict with 'text', 'prompt_tokens', 'completion_tokens'
        """
        if self._api_base:
            return self._generate_api(messages, temperature, max_new_tokens)
        self._ensure_loaded()

        import torch

        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # disable Qwen3 thinking mode
        )

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                prefill_chunk_size=prefill_chunk_size,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        completion_tokens = len(new_tokens)
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Free GPU memory from inference intermediates
        del new_tokens, outputs, inputs
        torch.cuda.empty_cache()

        return {
            "text": response.strip(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
