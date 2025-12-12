# src/tools/test_prompt.py
"""
Quick test harness to inspect raw HF pipeline output for a sample prompt.
Run: python -m src.tools.test_prompt
"""

from __future__ import annotations
import os, json, textwrap
from pathlib import Path

# Try to import your make_llm factory (which you already placed in src/tools/llm_tools.py)
try:
    from .llm_tools import make_llm
except Exception:
    # fallback to attempt direct HF import
    make_llm = None

# Use the parser prompt that your pipeline uses
PARSER_PROMPT = textwrap.dedent(
    """You are a strict JSON normalizer. Input: a noisy product JSON string.
Output: a JSON object with EXACT keys: product_name, concentration, skin_type (list), key_ingredients (list),
benefits (list), how_to_use, side_effects, price. Return ONLY valid JSON.

Example output must start with { and end with }.

Input JSON:
{"product_name":"GlowBoost Vitamin C Serum","concentration":"10% Vitamin C","skin_type":["Oily","Combination"],
"key_ingredients":["Vitamin C","Hyaluronic Acid"],"benefits":["Brightening","Fades dark spots"],
"how_to_use":"Apply 2-3 drops in the morning before sunscreen","side_effects":"Mild tingling for sensitive skin","price":"₹699"}
"""
)

# helper: call llm and print normalized results
def run_via_make_llm():
    if make_llm is None:
        print("make_llm() not available to import from src.tools.llm_tools. Skipping this path.")
        return
    llm = make_llm()
    print("LLM object:", type(llm))
    # if HF wrapper callable (our tool sets ._gen_defaults), call it directly
    if callable(llm):
        print("Calling HF wrapper (callable) ...")
        # try several generation kwargs to debug
        scenarios = [
            {"max_new_tokens": 200, "temperature": 0.0, "do_sample": False},
            {"max_new_tokens": 256, "temperature": 0.0, "do_sample": True},
            {"max_new_tokens": 512, "temperature": 0.2, "do_sample": True},
        ]
        for i, kw in enumerate(scenarios, start=1):
            print(f"\n--- Scenario {i}: {kw} ---")
            try:
                out = llm(PARSER_PROMPT, **kw)
                print("Raw pipeline output type:", type(out))
                print("Raw output (repr):")
                print(repr(out[:2] if isinstance(out, list) else out))
                # try to extract generated_texts
                if isinstance(out, list):
                    for j, it in enumerate(out[:3]):
                        print("Item", j, "keys:", list(it.keys()))
                        print("generated_text (first 1000 chars):")
                        print(it.get("generated_text", "")[:1000])
                else:
                    print("Direct output str:", str(out)[:2000])
            except Exception as e:
                print("Exception during pipeline call:", e)
    else:
        # assume LangChain ChatOpenAI object
        print("LLM appears to be a LangChain LLM:", llm)
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            tmp = PromptTemplate(input_variables=["product_json"], template="{product_json}")
            chain = LLMChain(llm=llm, prompt=tmp)
            res = chain.run(product_json=PARSER_PROMPT[:2000])
            print("LangChain LLMChain returned:", res[:2000])
        except Exception as e:
            print("LangChain run failed:", e)

def run_direct_hf_pipeline():
    try:
        from transformers import pipeline, AutoTokenizer
    except Exception as e:
        print("transformers import failed:", e)
        return
    model_name = os.getenv("LOCAL_MODEL_NAME", "google/flan-t5-small")
    task = os.getenv("HF_PIPELINE_TASK", "text2text-generation")
    device = int(os.getenv("HF_DEVICE", "-1"))
    print(f"Creating direct HF pipeline (model={model_name}, task={task}, device={device})")
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        print("tokenizer load warning:", e); tok = None
    try:
        pipe = pipeline(task, model=model_name, tokenizer=tok, device=device)
    except Exception as e:
        print("pipeline() creation failed:", e); return

    gen_kwargs = {
        "max_new_tokens": int(os.getenv("HF_MAX_NEW_TOKENS", "256")),
        "temperature": float(os.getenv("HF_TEMPERATURE", "0.0")),
        "top_p": float(os.getenv("HF_TOP_P", "0.95")),
        "do_sample": os.getenv("HF_DO_SAMPLE", "0") in ("1", "true", "True"),
    }

    print("Calling pipeline with gen_kwargs:", gen_kwargs)
    try:
        out = pipe(PARSER_PROMPT, **gen_kwargs)
        print("Raw out type:", type(out))
        if isinstance(out, list):
            for i, it in enumerate(out[:3]):
                print(f"OUT[{i}] keys:", it.keys())
                print("generated_text sample (first 2000 chars):")
                print(it.get("generated_text", "")[:2000])
        else:
            print("Out:", out)
    except Exception as e:
        print("Pipeline generation failed:", e)

if __name__ == "__main__":
    print("ENV vars snapshot:")
    for k in ("USE_LOCAL_LLM","LOCAL_MODEL_NAME","HF_PIPELINE_TASK","HF_DEVICE","HF_MAX_NEW_TOKENS","HF_TEMPERATURE","HF_DO_SAMPLE"):
        print(f"  {k} = {os.getenv(k)}")
    print("\nTrying make_llm() path:")
    run_via_make_llm()
    print("\nTrying direct HF pipeline path:")
    run_direct_hf_pipeline()
