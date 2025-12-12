"""
src/tools/llm_tools.py

Provides make_llm() which returns an LLM object suitable for:
 - LangChain (ChatOpenAI or HuggingFacePipeline) when available
 - OR a callable wrapper around a HF pipeline (used by existing fallback code)

Environment variables:
 - USE_LOCAL_LLM (1/0)     -> prefer a local HF pipeline when "1"
 - LOCAL_MODEL_NAME       -> e.g. "google/flan-t5-small" or "distilgpt2"
 - HF_PIPELINE_TASK       -> "text2text-generation" for T5, "text-generation" for GPT-style
 - HF_DEVICE              -> "-1" (CPU) or "0" (first GPU) or "cuda:0"
 - HF_MAX_NEW_TOKENS      -> e.g. "512"
 - OPENAI_API_KEY         -> used if not using local
 - DEBUG_RAW (optional)
"""

from __future__ import annotations
import os
import warnings
from typing import Any, Callable, Dict, Optional

# Try to import langchain OpenAI and wrappers (optional)
HAS_LANGCHAIN = False
try:
    from langchain.chat_models import ChatOpenAI
    from langchain.llms import HuggingFacePipeline as LC_HuggingFacePipeline
    HAS_LANGCHAIN = True
except Exception:
    # We'll still support non-LangChain fallback
    ChatOpenAI = None
    LC_HuggingFacePipeline = None
    HAS_LANGCHAIN = False

# Transformers (for local HF pipeline)
HAS_TRANSFORMERS = False
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except Exception:
    pipeline = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    AutoModelForCausalLM = None
    HAS_TRANSFORMERS = False

# small cache so we don't re-init repeatedly
_cached_pipeline = None
_cached_langchain_llm = None


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name, default)
    return v


def _parse_bool_env(name: str, default: bool = False) -> bool:
    v = _env(name)
    if v is None:
        return default
    return str(v).strip().lower() not in ("0", "false", "no", "")


def _get_hf_gen_kwargs() -> Dict[str, Any]:
    max_new = int(_env("HF_MAX_NEW_TOKENS", "256"))
    # Use deterministic defaults; user can change via environment
    return {"max_new_tokens": max_new, "do_sample": False, "temperature": 0.0, "top_p": 0.95}


def _create_hf_pipeline(model_name: str, task: str, device: str) -> Any:
    """
    Create and cache a transformers pipeline.
    Returns the pipeline object.
    """
    global _cached_pipeline
    if _cached_pipeline is not None and getattr(_cached_pipeline, "model_name", None) == model_name:
        return _cached_pipeline

    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers not installed. pip install transformers torch accelerate (or use OpenAI)")

    device_map = -1
    # device can be "-1" or "0" or "cuda:0"
    try:
        if str(device).startswith("cuda") or str(device).startswith("0"):
            # allow "0" or "cuda:0"
            if str(device) == "-1":
                device_map = -1
            elif ":" in str(device) and "cuda" in str(device):
                # transformers pipeline device accepts int GPU id, 0 for cuda:0
                device_map = int(str(device).split(":")[-1])
            else:
                device_map = int(device)
        else:
            device_map = -1
    except Exception:
        device_map = -1

    hf_task = task or "text-generation"

    # Auto-select model class for better tokenizer/model compatibility
    try:
        if hf_task == "text2text-generation":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            # causal
            model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception:
        # fallback: pipeline will fetch model internally
        model = None

    try:
        # create the pipeline (this will download model if needed)
        pl = pipeline(task=hf_task, model=model_name if model is None else model, device=device_map, tokenizer=model_name)
    except Exception:
        # simpler pipeline creation (often works)
        pl = pipeline(task=hf_task, model=model_name, device=device_map)

    # attach meta so caller can inspect
    pl.model_name = model_name
    pl._gen_defaults = _get_hf_gen_kwargs()
    _cached_pipeline = pl
    return pl


class _CallableWrapper:
    """
    Wrap a HF pipeline into a callable with attribute "_gen_defaults" to be compatible
    with existing code that expects llm to be callable and may read llm._gen_defaults.
    """
    def __init__(self, pl):
        self._pl = pl
        # allow code to read generation defaults
        self._gen_defaults = getattr(pl, "_gen_defaults", {})

    def __call__(self, prompt: str, **kwargs) -> Any:
        # merge defaults & kwargs
        gen_kwargs = dict(self._gen_defaults or {})
        gen_kwargs.update(kwargs or {})
        # transformers pipeline expects either text or a list
        out = self._pl(prompt, **gen_kwargs)
        return out

    # small convenience to mimic LangChain LLM when not available
    def generate(self, prompt_messages: Any, **kwargs):
        """
        Accepts a list of messages or strings; returns a simple wrapper object similar to LangChain's return shape.
        We'll produce a minimal object so LLMChain.generate may work when passed this wrapper.
        """
        texts = []
        # If prompt_messages is LangChain Messages, join them; if it's strings, use first
        try:
            # accept dict/list of dicts or strings
            if isinstance(prompt_messages, list):
                # try to extract text if elements are dict-like
                for m in prompt_messages:
                    if isinstance(m, dict):
                        texts.append(m.get("content") or m.get("text") or str(m))
                    else:
                        texts.append(str(m))
            else:
                texts = [str(prompt_messages)]
        except Exception:
            texts = [str(prompt_messages)]
        joined = "\n".join(texts)
        out = self.__call__(joined)
        # normalize output to langchain-like .generations structure if possible
        if isinstance(out, list) and len(out) > 0:
            return {"generations": [[{"text": out[0].get("generated_text", str(out[0])) if isinstance(out[0], dict) else str(out[0])}]]}
        else:
            return {"generations": [[{"text": str(out)}]]}


def make_llm() -> Any:
    """
    Construct an LLM or pipeline object based on environment.

    Returns:
      - If OpenAI & LangChain available and USE_LOCAL_LLM not set: a ChatOpenAI instance
      - Else if local HF requested & transformers available:
         - If LangChain present -> return a LangChain HuggingFacePipeline object wrapping a transformers pipeline
         - Else -> return a callable wrapper around transformers.pipeline with attribute _gen_defaults (used by existing code)
    """
    global _cached_langchain_llm

    use_local = _parse_bool_env("USE_LOCAL_LLM", False)
    if use_local:
        model_name = _env("LOCAL_MODEL_NAME", "distilgpt2")
        hf_task = _env("HF_PIPELINE_TASK", "text-generation")
        device = _env("HF_DEVICE", "-1")
        # Create HF pipeline
        pl = _create_hf_pipeline(model_name, hf_task, device)
        # If LangChain is available, wrap in LangChain's HuggingFacePipeline for compatibility with AgentExecutor/LMMChains
        if HAS_LANGCHAIN and LC_HuggingFacePipeline is not None:
            # Build a langchain HuggingFacePipeline wrapper
            hf_wrapper = LC_HuggingFacePipeline(pipeline=pl)
            # LangChain wrapper expects .generate and behaves like an LLM
            _cached_langchain_llm = hf_wrapper
            return hf_wrapper
        # else return callable wrapper
        return _CallableWrapper(pl)

    # Not using local -> prefer OpenAI via LangChain if available
    openai_key = _env("OPENAI_API_KEY", None)
    if openai_key and HAS_LANGCHAIN and ChatOpenAI is not None:
        # create ChatOpenAI wrapper (use env for model name if present)
        model_name = _env("OPENAI_MODEL", "gpt-4o")  # default to something — user can set OPENAI_MODEL
        # convert temperature/max tokens from env if present
        temperature = float(_env("OPENAI_TEMPERATURE", "0.0"))
        # ChatOpenAI will pick up OPENAI_API_KEY from env automatically
        llm = ChatOpenAI(model=model_name, temperature=temperature)
        _cached_langchain_llm = llm
        return llm

    # If we reach here, we don't have OpenAI/LangChain or local HF; try to fallback to HF pipeline if transformers installed
    if HAS_TRANSFORMERS:
        # fallback to local distilgpt2
        model_name = _env("LOCAL_MODEL_NAME", "distilgpt2")
        hf_task = _env("HF_PIPELINE_TASK", "text-generation")
        device = _env("HF_DEVICE", "-1")
        pl = _create_hf_pipeline(model_name, hf_task, device)
        return _CallableWrapper(pl)

    raise RuntimeError("No LLM available: set OPENAI_API_KEY or install transformers and set USE_LOCAL_LLM=1")


# Optional: quick test helper (callable from CLI)
if __name__ == "__main__":
    print("Testing make_llm() with current environment...")
    try:
        llm = make_llm()
        print("LLM object type:", type(llm))
        # show _gen_defaults if any
        gd = getattr(llm, "_gen_defaults", None) or getattr(getattr(llm, "pipeline", None), "_gen_defaults", None)
        print("Generation defaults:", gd)
    except Exception as e:
        print("make_llm() error:", e)
        raise
