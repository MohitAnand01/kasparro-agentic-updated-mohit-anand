# src/agents/langchain_agent_system.py
"""
LangChain "agentic" system (runtime-created PromptTemplates + LLMChain orchestration).

This file provides run_agent_system(use_mock: bool, outputs_dir: str).
- It builds PromptTemplates at runtime (no pydantic validation errors at import time).
- If LangChain is available and make_llm() returns a LangChain LLM, it runs via LLMChain steps.
- If make_llm() returns a local HF pipeline (callable), it will run the same steps using plain
  string prompts and the callable interface.
- On any failure, it falls back to deterministic pipeline in src.agents.langchain_pipeline.
"""

from __future__ import annotations
import os
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import LangChain objects lazily
HAS_LANGCHAIN = False
try:
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    HAS_LANGCHAIN = True
except Exception:
    PromptTemplate = None
    LLMChain = None
    HAS_LANGCHAIN = False

# Import helper utilities and fallback pipeline runner
try:
    from src.tools.file_tools import load_product_json_text, write_json_file, ensure_outputs_dir
    from src.tools.llm_tools import make_llm
    from src.agents.langchain_pipeline import _safe_load_json, _maybe_debug, _sanitize_faqs, _normalize_text
    # use pipeline fallback function if needed
    from src.agents.langchain_pipeline import run_pipeline as deterministic_pipeline
except Exception:
    # Minimal fallback helpers if the repo helpers are missing (shouldn't happen in normal repo)
    def load_product_json_text(path: str) -> Optional[str]:
        p = Path(path)
        return p.read_text(encoding="utf-8") if p.exists() else None

    def ensure_outputs_dir(path: str = "outputs"):
        Path(path).mkdir(parents=True, exist_ok=True)

    def write_json_file(fname: str, data: Any, outputs_dir: str = "outputs"):
        ensure_outputs_dir(outputs_dir)
        outp = Path(outputs_dir) / fname
        outp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Wrote {outp}")

    def make_llm():
        raise RuntimeError("tools.llm_tools.make_llm() not found; restore file.")

    def _safe_load_json(x, fallback=None):
        try:
            return json.loads(x)
        except Exception:
            return fallback

    def _maybe_debug(stage, raw):
        if str(os.getenv("DEBUG_RAW", "")).lower() in ("1", "true", "yes"):
            print(f"\n--- RAW OUTPUT ({stage}) ---\n")
            print(raw or "<EMPTY>")
            print(f"\n--- END RAW OUTPUT ({stage}) ---\n")

    def _sanitize_faqs(faqs):
        return faqs or []

    def _normalize_text(t):
        return (t or "").strip().lower()

# Prompt texts (use double braces where we want literal braces in string)
PARSER_PROMPT_TEXT = (
    "You are a strict JSON normalizer. Input: a noisy product JSON string.\n"
    "Output: a normalized JSON object with keys exactly: product_name, concentration, skin_type (list), "
    "key_ingredients (list), benefits (list), how_to_use, side_effects, price. Return ONLY valid JSON.\n\n"
    "Input JSON:\n{product_json}\n\nONLY output valid JSON and nothing else. Start with '{' and end with '}'."
)

QGEN_PROMPT_TEXT = (
    "You are a product-question generator. Given the product JSON below, produce a JSON array "
    "of at least 10 objects. Each object MUST be of the exact form: "
    '{{"question":"...","category":"Informational|Usage|Safety|Purchase|Comparison"}}.\n'
    "Return ONLY a valid JSON array (no commentary). Start with '[' and end with ']'.\n\nProduct JSON:\n{product_json}"
)

PLANNER_PROMPT_TEXT = (
    "You are an assistant that writes helpful FAQ answers using ONLY the product facts.\n"
    "Given product JSON:\n{product_json}\n\nAnd the question:\n{question}\n\n"
    "Produce a JSON object: {{\"question\":\"...\",\"answer\":\"...\",\"category\":\"...\"}}\n"
    "Answer must be concise (1-3 sentences), factual (use product fields), and return ONLY valid JSON."
)

ASSEMBLER_PROMPT_TEXT = (
    "Assemble a final product page JSON object using the product JSON and the FAQ items.\n\n"
    "Product JSON:\n{product_json}\n\nFAQ items JSON array:\n{faqs_json_str}\n\n"
    "Return ONE JSON object with keys: title, name, concentration, skin_types, key_ingredients, "
    "benefits_summary (list), usage_instructions, safety_information, price, faqs (the provided list). "
    "Return ONLY valid JSON. Start with '{' and end with '}'."
)

# Helper to run a string prompt via either LangChain LLMChain or a HF pipeline callable
def _run_prompt(llm, template_text: str, input_vars: Dict[str, Any]) -> str:
    """
    llm: could be:
      - a LangChain LLM (object supported by LLMChain) -> we build a PromptTemplate and run LLMChain
      - a callable (local HF pipeline) -> we format the template_text and call llm(prompt_text, **gen_kwargs)
    """
    # LangChain path (llm compatible with LLMChain)
    if HAS_LANGCHAIN and PromptTemplate is not None and LLMChain is not None and hasattr(llm, "generate"):
        prompt = PromptTemplate(input_variables=list(input_vars.keys()), template=template_text)
        chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
        return chain.run(**input_vars)

    # Callable HF pipeline path
    if callable(llm):
        prompt_text = template_text.format(**input_vars)
        gen_kwargs = getattr(llm, "_gen_defaults", {})
        out = llm(prompt_text, **gen_kwargs)
        # HF pipelines return list of dicts
        if isinstance(out, list) and len(out) > 0:
            first = out[0]
            if isinstance(first, dict) and "generated_text" in first:
                return first["generated_text"]
            return str(first)
        return str(out)

    raise RuntimeError("Unsupported LLM type in _run_prompt()")

# Small utility to attempt parsing and fallback gracefully
def _try_parse_json(raw: str, stage: str = "stage"):
    _maybe_debug(stage, raw)
    parsed = _safe_load_json(raw, fallback=None)
    return parsed

# The main exposed function
def run_agent_system(use_mock: bool = False, outputs_dir: str = "outputs"):
    """
    Top-level entry for agentic system.
    Attempts to use LangChain LLMChains at runtime; if anything fails, falls back to the deterministic pipeline.
    """
    ensure_outputs_dir(outputs_dir)

    # Load product input
    raw_text = load_product_json_text("src/data/product_input.json")
    if not raw_text:
        print("❌ src/data/product_input.json missing; falling back to deterministic pipeline.")
        return deterministic_pipeline(use_mock=True, outputs_dir=outputs_dir)

    # If mock requested just use deterministic pipeline (keeps behavior consistent)
    if use_mock:
        print("🔷 Running deterministic fallback (mock mode).")
        return deterministic_pipeline(use_mock=True, outputs_dir=outputs_dir)

    # Try to create LLM via make_llm()
    try:
        llm = make_llm()
        print("🔗 LLM initialized.")
    except Exception as e:
        print("⚠️ make_llm() failed:", e)
        print("➡️ Falling back to deterministic pipeline.")
        return deterministic_pipeline(use_mock=True, outputs_dir=outputs_dir)

    # At this point we have llm — may be LangChain LLM or a callable HF pipeline.
    # We'll run four LLM steps: parser, qgen, planner (loop), assembler.
    try:
        # Stage 1: Parser -> get normalized product
        parser_out = _run_prompt(llm, PARSER_PROMPT_TEXT, {"product_json": raw_text})
        parsed_product = _try_parse_json(parser_out, "PARSER")
        if not isinstance(parsed_product, dict):
            print("⚠️ Parser produced invalid JSON — using deterministic pipeline fallback.")
            return deterministic_pipeline(use_mock=True, outputs_dir=outputs_dir)

        print("📘 Parsed product keys:", list(parsed_product.keys()))

        # Stage 2: Question generation
        qgen_out = _run_prompt(llm, QGEN_PROMPT_TEXT, {"product_json": json.dumps(parsed_product, ensure_ascii=False)})
        questions = _try_parse_json(qgen_out, "QGEN")
        if not isinstance(questions, list):
            print("⚠️ QGen returned non-list — falling back to canned questions.")
            # fallback canned questions (simple deterministic list)
            questions = [
                {"question": "What is the product used for?", "category": "Informational"},
                {"question": "What are the key ingredients?", "category": "Informational"},
                {"question": "How should I use it?", "category": "Usage"},
                {"question": "Any side effects?", "category": "Safety"},
                {"question": "What is the price?", "category": "Purchase"},
            ]

        print("💬 Questions count:", len(questions))

        # Stage 3: Planner — build answers (call planner per question)
        faqs: List[Dict[str, Any]] = []
        for idx, q in enumerate(questions):
            if isinstance(q, dict) and q.get("question"):
                qtext = q["question"]
                planner_out = _run_prompt(llm, PLANNER_PROMPT_TEXT, {"product_json": json.dumps(parsed_product, ensure_ascii=False), "question": qtext})
                item = _try_parse_json(planner_out, f"PLANNER_q{idx+1}")
                if not isinstance(item, dict) or "answer" not in item:
                    # fallback deterministic answer using product facts
                    item = {"question": qtext, "answer": parsed_product.get("how_to_use") or "See product details.", "category": q.get("category", "Informational")}
                faqs.append(item)
        print("🧩 Built FAQs count:", len(faqs))

        # Stage 4: Assembler
        # Trim faqs to assembler limit
        FAQ_TRIM_N = int(os.getenv("ASSEMBLER_FAQ_LIMIT", "8"))
        faqs_trimmed = faqs[:FAQ_TRIM_N]
        assembler_input_product = {
            "product_name": parsed_product.get("product_name"),
            "concentration": parsed_product.get("concentration"),
            "skin_type": parsed_product.get("skin_type"),
            "key_ingredients": parsed_product.get("key_ingredients"),
            "benefits": parsed_product.get("benefits"),
            "how_to_use": parsed_product.get("how_to_use"),
            "side_effects": parsed_product.get("side_effects"),
            "price": parsed_product.get("price"),
        }
        assembler_out = _run_prompt(llm, ASSEMBLER_PROMPT_TEXT, {"product_json": json.dumps(assembler_input_product, ensure_ascii=False), "faqs_json_str": json.dumps(faqs_trimmed, ensure_ascii=False)})
        product_page = _try_parse_json(assembler_out, "ASSEMBLER")
        if not isinstance(product_page, dict):
            print("⚠️ Assembler returned invalid JSON; using deterministic assembly fallback.")
            # create a deterministic product page
            product_page = {
                "title": parsed_product.get("product_name"),
                "name": parsed_product.get("product_name"),
                "concentration": parsed_product.get("concentration"),
                "skin_types": parsed_product.get("skin_type"),
                "key_ingredients": parsed_product.get("key_ingredients"),
                "benefits_summary": parsed_product.get("benefits"),
                "usage_instructions": parsed_product.get("how_to_use"),
                "safety_information": parsed_product.get("side_effects"),
                "price": parsed_product.get("price"),
            }

        # Sanitize FAQs and attach to product_page
        cleaned_faqs = _sanitize_faqs(faqs)
        product_page["faqs"] = cleaned_faqs

        # Simple comparison page
        comparison_page = {
            "base": {"name": parsed_product.get("product_name"), "price": parsed_product.get("price")},
            "competitor": {"name": "Fictional B", "price": "₹1,199"},
        }

        # FAQ page
        faq_page = {"product_name": parsed_product.get("product_name"), "faqs": cleaned_faqs}

        # Write outputs
        write_json_file("product_page.json", product_page, outputs_dir=outputs_dir)
        write_json_file("faq.json", faq_page, outputs_dir=outputs_dir)
        write_json_file("comparison_page.json", comparison_page, outputs_dir=outputs_dir)

        print("📁 Wrote outputs:", ["product_page.json", "faq.json", "comparison_page.json"])
        print("✅ Agentic-style pipeline finished.")
        return

    except Exception as e:
        print("⚠️ Agentic pipeline failed with an exception; falling back to deterministic pipeline.")
        traceback.print_exc()
        return deterministic_pipeline(use_mock=True, outputs_dir=outputs_dir)


# Allow running directly for quick testing
if __name__ == "__main__":
    use_mock_env = os.getenv("USE_MOCK", "1")
    use_mock_flag = use_mock_env.strip() not in ("0", "false", "False")
    run_agent_system(use_mock=use_mock_flag, outputs_dir="outputs")

