"""
src/agents/langchain_pipeline.py

Finalized pipeline. Safe to run with either:
 - USE_MOCK=1 (deterministic mock path)
 - USE_LOCAL_LLM=1 (local HF pipeline via make_llm)
 - default OpenAI via make_llm (requires OPENAI_API_KEY)

This file intentionally creates PromptTemplate objects at runtime so import-time
pydantic errors do not occur (LangChain's PromptTemplate sometimes validates schema
on init).
"""

from __future__ import annotations
import os
import json
import re
import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable

# runtime-safe LangChain imports
HAS_LANGCHAIN = False
try:
    from langchain.prompts import PromptTemplate  # type: ignore
    from langchain.chains import LLMChain  # type: ignore
    HAS_LANGCHAIN = True
except Exception:
    PromptTemplate = None  # type: ignore
    LLMChain = None  # type: ignore
    HAS_LANGCHAIN = False

# local helpers
try:
    from tools.file_tools import load_product_json_text, write_json_file, ensure_outputs_dir  # type: ignore
    from tools.llm_tools import make_llm  # type: ignore
except Exception:
    # Minimal fallbacks if those modules are missing; these should be replaced by your repo files.
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

# ---------- Prompt texts (templates are created at runtime to avoid Pydantic validation issues)
PARSER_PROMPT_TEXT = (
    "You are a strict JSON normalizer. Input: a noisy product JSON string. "
    "Output: a normalized JSON object with keys exactly: product_name, concentration, skin_type (list), "
    "key_ingredients (list), benefits (list), how_to_use, side_effects, price. Return ONLY valid JSON.\n\n"
    "Input JSON:\n{product_json}\n\nONLY output valid JSON and nothing else. Start with '{' and end with '}'."
)

QGEN_PROMPT_TEXT = (
    "You are a product-question generator. Given the product JSON below, produce a JSON array "
    "of at least 15 objects. Each object MUST be of the exact form: "
    "{\"question\":\"...\",\"category\":\"Informational|Usage|Safety|Purchase|Comparison\"}.\n"
    "Return ONLY a valid JSON array (no commentary). Start with '[' and end with ']'.\n\nProduct JSON:\n{product_json}"
)

PLANNER_PROMPT_TEXT = (
    "You are an assistant that writes helpful FAQ answers using ONLY the product facts. "
    "Given product JSON:\n{product_json}\n\nAnd the question:\n{question}\n\n"
    "Produce a JSON object: {\"question\":\"...\",\"answer\":\"...\",\"category\":\"...\"}\n"
    "Answer must be concise (1-2 sentences), factual (use product fields), and return ONLY valid JSON."
)

ASSEMBLER_PROMPT_TEXT = (
    "Assemble a final product page JSON object using the product JSON and the FAQ items.\n\n"
    "Product JSON:\n{product_json}\n\nFAQ items JSON array:\n{faqs_json_str}\n\n"
    "Return ONE JSON object with keys: title, name, concentration, skin_types, key_ingredients, "
    "benefits_summary (list), usage_instructions, safety_information, price, faqs (the provided list). "
    "Return ONLY valid JSON. Start with '{' and end with '}'."
)

# ---------- helpers for JSON extraction + sanitization
def _extract_json_snippet(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    s = text.strip()
    try:
        json.loads(s)
        return s
    except Exception:
        pass
    # Balanced braces/brackets search
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = s.find(open_ch)
        while start != -1:
            depth = 0
            for i in range(start, len(s)):
                ch = s[i]
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        candidate = s[start : i + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except Exception:
                            break
            start = s.find(open_ch, start + 1)
    # fallback regex (less reliable)
    m = re.search(r"(\{(?:.|\n)*\}|\[(?:.|\n)*\])", s, flags=re.DOTALL)
    if m:
        candidate = m.group(1)
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            return None
    return None

def _safe_load_json(text: Optional[str], fallback=None):
    if text is None:
        return fallback
    s = text.strip()
    try:
        return json.loads(s)
    except Exception:
        snippet = _extract_json_snippet(s)
        if snippet:
            try:
                return json.loads(snippet)
            except Exception:
                return fallback
        return fallback

def _maybe_debug(stage_name: str, raw_text: Optional[str]):
    if str(os.getenv("DEBUG_RAW", "")).lower() in ("1", "true", "yes"):
        print(f"\n--- RAW OUTPUT ({stage_name}) ---\n")
        if raw_text is None:
            print("<EMPTY>\n")
        else:
            print(raw_text[:5000])
            if len(raw_text) > 5000:
                print("\n... (truncated)\n")
        print(f"--- END RAW OUTPUT ({stage_name}) ---\n")

def _normalize_text(t: Optional[str]) -> str:
    if t is None:
        return ""
    s = str(t)
    s = html.unescape(s)
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = " ".join(s.split())
    return s.strip().lower()

def _sanitize_faqs(faqs: Iterable[dict]) -> List[dict]:
    clean: List[dict] = []
    seen = set()
    for item in faqs or []:
        if not isinstance(item, dict):
            continue
        q = item.get("question") or item.get("q") or None
        a = item.get("answer") or item.get("ans") or None
        c = item.get("category") or "Informational"
        if q is None:
            continue
        nq = _normalize_text(q)
        if not nq or nq in seen:
            continue
        if a is None:
            a = "See product details."
        clean_item = {"question": str(q).strip(), "answer": str(a).strip(), "category": str(c).strip() if c else "Informational"}
        clean.append(clean_item)
        seen.add(nq)
    return clean

# ---------- LLM runner (supports LangChain LLMs and raw HF pipelines)
def _run_prompt_with_llm(llm, template_obj_or_text, **vars) -> str:
    # LangChain path: if LangChain available and template_obj_or_text is PromptTemplate
    if HAS_LANGCHAIN and PromptTemplate is not None and isinstance(template_obj_or_text, PromptTemplate) and hasattr(llm, "generate"):
        chain = LLMChain(llm=llm, prompt=template_obj_or_text, verbose=False)
        return chain.run(**vars)

    # Raw HF pipeline callable path: expect llm to be callable and template_obj_or_text a string template
    if isinstance(template_obj_or_text, str) and callable(llm):
        prompt_text = template_obj_or_text.format(**vars)
        gen_kwargs = getattr(llm, "_gen_defaults", {})
        out = llm(prompt_text, **(gen_kwargs or {}))
        # transformers pipeline returns list of dicts for generation; pick generated_text
        if isinstance(out, list) and len(out) > 0:
            first = out[0]
            if isinstance(first, dict):
                return first.get("generated_text", "") or str(first)
            return str(first)
        return str(out)

    # Fallback: if LangChain exists, create PromptTemplate on the fly
    if isinstance(template_obj_or_text, str) and HAS_LANGCHAIN and PromptTemplate is not None:
        tmp = PromptTemplate(input_variables=list(vars.keys()), template=template_obj_or_text)
        chain = LLMChain(llm=llm, prompt=tmp, verbose=False)
        return chain.run(**vars)

    raise RuntimeError("Unsupported LLM + template combination in _run_prompt_with_llm()")

# ---------- main pipeline
def run_pipeline(use_mock: bool = False, outputs_dir: str = "outputs"):
    ensure_outputs_dir(outputs_dir)

    # Some environments may want to skip agent attempts entirely
    use_agent = str(os.getenv("USE_AGENT", "1")).strip().lower() not in ("0", "false", "no")
    if not use_agent:
        print("▶ USE_AGENT=0 — skipping agent system/AgentExecutor attempts (deterministic pipeline only).")

    # Build LLM if not mock
    llm = None
    if not use_mock:
        try:
            llm = make_llm()
            print("🔗 LLM initialized.")
        except Exception as e:
            print("⚠️ make_llm() failed:", e)
            print("➡️ Falling back to mock mode.")
            use_mock = True

    # Create PromptTemplate objects at runtime if LangChain available
    parser_prompt_obj = qgen_prompt_obj = planner_prompt_obj = assembler_prompt_obj = None
    if not use_mock and HAS_LANGCHAIN and PromptTemplate is not None:
        parser_prompt_obj = PromptTemplate(input_variables=["product_json"], template=PARSER_PROMPT_TEXT)
        qgen_prompt_obj = PromptTemplate(input_variables=["product_json"], template=QGEN_PROMPT_TEXT)
        planner_prompt_obj = PromptTemplate(input_variables=["product_json", "question"], template=PLANNER_PROMPT_TEXT)
        assembler_prompt_obj = PromptTemplate(input_variables=["product_json", "faqs_json_str"], template=ASSEMBLER_PROMPT_TEXT)

    # Load dataset
    raw_text = load_product_json_text("src/data/product_input.json")
    if not raw_text:
        raise FileNotFoundError("src/data/product_input.json not found or empty. Place dataset file there.")

    # Stage 1: Parser
    if use_mock:
        parsed_product = _mock_parse(raw_text)
    else:
        try:
            parser_raw = _run_prompt_with_llm(llm, parser_prompt_obj or PARSER_PROMPT_TEXT, product_json=raw_text)
            _maybe_debug("PARSER", parser_raw)
            parsed_product = _safe_load_json(parser_raw, fallback=None)
            if parsed_product is None or not isinstance(parsed_product, dict):
                print("⚠️ Parser produced invalid JSON — using deterministic fallback.")
                parsed_product = _mock_parse(raw_text)
        except Exception as e:
            print("⚠️ Parser error:", e)
            parsed_product = _mock_parse(raw_text)
            use_mock = True

    print("📘 Parsed product type ->", type(parsed_product).__name__)

    # Stage 2: Question generation
    if use_mock:
        questions = _mock_questions(parsed_product)
    else:
        try:
            qgen_raw = _run_prompt_with_llm(llm, qgen_prompt_obj or QGEN_PROMPT_TEXT, product_json=json.dumps(parsed_product, ensure_ascii=False))
            _maybe_debug("QGEN", qgen_raw)
            questions = _safe_load_json(qgen_raw, fallback=None)
            if not isinstance(questions, list):
                print("⚠️ QGen returned non-list — falling back to canned questions.")
                questions = _mock_questions(parsed_product)
        except Exception as e:
            print("⚠️ QGen error:", e)
            questions = _mock_questions(parsed_product)
            use_mock = True

    print("💬 Generated questions count:", len(questions))

    # Stage 3: Planner -> build FAQs
    faqs: List[Dict[str, Any]] = []
    if use_mock:
        faqs = _mock_build_faqs(parsed_product, questions)
    else:
        for idx, qitem in enumerate(questions):
            if isinstance(qitem, dict) and "question" in qitem:
                q_text = qitem["question"]
                try:
                    planner_raw = _run_prompt_with_llm(llm, planner_prompt_obj or PLANNER_PROMPT_TEXT, product_json=json.dumps(parsed_product, ensure_ascii=False), question=q_text)
                    _maybe_debug(f"PLANNER_q{idx+1}", planner_raw)
                    item = _safe_load_json(planner_raw, fallback=None)
                    if not item or not isinstance(item, dict):
                        item = {"question": q_text, "answer": f"See product: {parsed_product.get('product_name')}.", "category": qitem.get("category", "Informational")}
                except Exception as e:
                    print("⚠️ Planner exception for question:", q_text, "->", e)
                    item = {"question": q_text, "answer": f"See product: {parsed_product.get('product_name')}.", "category": qitem.get("category", "Informational")}
                faqs.append(item)
            else:
                continue

    print("🧩 Built FAQ items (raw count):", len(faqs))

    # Stage 4: Assembler (trim inputs to avoid encoder overflow)
    assembler_product = {
        "product_name": parsed_product.get("product_name"),
        "concentration": parsed_product.get("concentration"),
        "skin_type": parsed_product.get("skin_type"),
        "key_ingredients": parsed_product.get("key_ingredients"),
        "benefits": parsed_product.get("benefits"),
        "how_to_use": parsed_product.get("how_to_use"),
        "side_effects": parsed_product.get("side_effects"),
        "price": parsed_product.get("price"),
    }
    FAQ_TRIM_N = int(os.getenv("ASSEMBLER_FAQ_LIMIT", "8"))
    faqs_for_assembler = (faqs or [])[:FAQ_TRIM_N]

    if use_mock:
        product_page = _mock_product_page(parsed_product, faqs)
        comparison_page = _mock_comparison_page(parsed_product)
        faq_page = {"product_name": parsed_product.get("product_name"), "faqs": faqs}
    else:
        try:
            assembler_raw = _run_prompt_with_llm(llm, assembler_prompt_obj or ASSEMBLER_PROMPT_TEXT, product_json=json.dumps(assembler_product, ensure_ascii=False), faqs_json_str=json.dumps(faqs_for_assembler, ensure_ascii=False))
            _maybe_debug("ASSEMBLER", assembler_raw)
            product_page = _safe_load_json(assembler_raw, fallback=None)
            if not product_page or not isinstance(product_page, dict):
                print("⚠️ Assembler returned invalid JSON (likely due to input size); using deterministic assembly.")
                product_page = _mock_product_page(parsed_product, faqs)
            comparison_page = _mock_comparison_page(parsed_product)
            faq_page = {"product_name": parsed_product.get("product_name"), "faqs": faqs}
        except Exception as e:
            print("⚠️ Assembler error:", e)
            product_page = _mock_product_page(parsed_product, faqs)
            comparison_page = _mock_comparison_page(parsed_product)
            faq_page = {"product_name": parsed_product.get("product_name"), "faqs": faqs}

    # Sanitize & dedupe FAQs before writing
    cleaned_faqs = _sanitize_faqs(faqs)
    if isinstance(product_page, dict):
        product_page["faqs"] = cleaned_faqs
    faq_page = {"product_name": parsed_product.get("product_name"), "faqs": cleaned_faqs}

    # Write outputs
    write_json_file("product_page.json", product_page, outputs_dir=outputs_dir)
    write_json_file("faq.json", faq_page, outputs_dir=outputs_dir)
    write_json_file("comparison_page.json", comparison_page, outputs_dir=outputs_dir)

    print("📁 Wrote outputs:", ["product_page.json", "faq.json", "comparison_page.json"])
    print("✅ Pipeline finished.")

# ---------- deterministic mocks (same as before)
def _mock_parse(raw_text: str) -> Dict[str, Any]:
    try:
        raw = json.loads(raw_text)
    except Exception:
        raw = {}
    return {
        "product_name": raw.get("product_name") or raw.get("name") or "GlowBoost Vitamin C Serum",
        "concentration": raw.get("concentration") or "10% Vitamin C",
        "skin_type": raw.get("skin_type") or raw.get("skin_types") or ["Oily", "Combination"],
        "key_ingredients": raw.get("key_ingredients") or raw.get("ingredients") or ["Vitamin C", "Hyaluronic Acid"],
        "benefits": raw.get("benefits") or ["Brightening", "Fades dark spots"],
        "how_to_use": raw.get("how_to_use") or "Apply 2–3 drops in the morning before sunscreen",
        "side_effects": raw.get("side_effects") or "Mild tingling for sensitive skin",
        "price": raw.get("price") or "₹699",
    }

def _mock_questions(parsed_product: Dict[str, Any]) -> List[Dict[str, str]]:
    base = parsed_product.get("product_name", "the product")
    questions = [
        {"question": f"What is {base} used for?", "category": "Informational"},
        {"question": "What are the key ingredients?", "category": "Informational"},
        {"question": "How often should I apply this serum?", "category": "Usage"},
        {"question": "Can this product cause irritation on sensitive skin?", "category": "Safety"},
        {"question": "What is the price and package size?", "category": "Purchase"},
        {"question": "How does this compare to other vitamin C serums?", "category": "Comparison"},
        {"question": "Should I use sunscreen after applying?", "category": "Usage"},
        {"question": "Is this suitable for oily skin?", "category": "Informational"},
        {"question": "Can I use this with retinol?", "category": "Safety"},
        {"question": "How long before I see results?", "category": "Informational"},
        {"question": "Does it contain fragrance or alcohol?", "category": "Safety"},
        {"question": "Is this product vegan/cruelty-free?", "category": "Informational"},
        {"question": "How should I store the product?", "category": "Usage"},
        {"question": "What packaging is used?", "category": "Purchase"},
        {"question": "Any known contraindications?", "category": "Safety"},
    ]
    return questions

def _mock_build_faqs(parsed_product: Dict[str, Any], qlist: List[Dict[str, str]]) -> List[Dict[str, str]]:
    faqs = []
    name = parsed_product.get("product_name", "The product")
    for q in qlist:
        question = q.get("question", "Question?")
        category = q.get("category", "Informational")
        answer = f"Mock answer derived from product: {name}."
        faqs.append({"question": question, "answer": answer, "category": category})
    return faqs

def _mock_product_page(parsed_product: Dict[str, Any], faqs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "title": parsed_product.get("product_name"),
        "name": parsed_product.get("product_name"),
        "concentration": parsed_product.get("concentration"),
        "skin_types": parsed_product.get("skin_type"),
        "key_ingredients": parsed_product.get("key_ingredients"),
        "benefits_summary": parsed_product.get("benefits"),
        "usage_instructions": parsed_product.get("how_to_use"),
        "safety_information": parsed_product.get("side_effects"),
        "price": parsed_product.get("price"),
        "faqs": faqs,
        "faqs_count": len(faqs),
    }

def _mock_comparison_page(parsed_product: Dict[str, Any]) -> Dict[str, Any]:
    competitor = {"name": "Fictional B (mock)", "price": "₹1,199", "key_ingredients": ["Vitamin C", "Niacinamide"]}
    return {
        "base_product_summary": {
            "name": parsed_product.get("product_name"),
            "price": parsed_product.get("price"),
            "key_ingredients": parsed_product.get("key_ingredients"),
        },
        "competitor": competitor,
        "comparison": f"Simple mock comparison between {parsed_product.get('product_name')} and {competitor['name']}.",
    }

if __name__ == "__main__":
    use_mock_env = os.getenv("USE_MOCK", "1")
    use_mock_flag = use_mock_env.strip() not in ("0", "false", "False")
    run_pipeline(use_mock=use_mock_flag, outputs_dir="outputs")
