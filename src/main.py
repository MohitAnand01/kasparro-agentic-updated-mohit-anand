# src/main.py
from __future__ import annotations
import os, sys, traceback

# Ensure project root (the directory that contains 'src') is on sys.path
THIS_FILE = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(THIS_FILE)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def _debug_list_src_packages():
    print("PYTHON sys.path (first 6 entries):")
    for p in sys.path[:6]:
        print("  ", p)
    print("\nFiles under src/:")
    for root, dirs, files in os.walk(SRC_DIR):
        level = root.replace(SRC_DIR, "").count(os.sep)
        indent = "  " * (level)
        print(f"{indent}{os.path.basename(root)}/")
        for f in files:
            print(f"{indent}  - {f}")
        # only show top levels to avoid huge output
        if level > 2:
            break

def main():
    print("🔷 Starting project runner (robust main.py)")
    _debug_list_src_packages()

    # env flags for behavior
    USE_AGENT = os.getenv("USE_AGENT", "1").strip().lower() not in ("0", "false", "no")
    USE_MOCK = os.getenv("USE_MOCK", "1").strip().lower() not in ("0", "false", "no")

    # attempt imports (late)
    run_agent_system = None
    run_pipeline = None

    try:
        from src.agents.langchain_agent_system import run_agent_system as _ras  # prefer explicit package import
        run_agent_system = _ras
        print("Imported src.agents.langchain_agent_system -> OK")
    except Exception as e:
        print("⚠️ Could not import src.agents.langchain_agent_system:", type(e).__name__, e)

    try:
        from src.agents.langchain_pipeline import run_pipeline as _rp
        run_pipeline = _rp
        print("Imported src.agents.langchain_pipeline -> OK")
    except Exception as e:
        print("⚠️ Could not import src.agents.langchain_pipeline:", type(e).__name__, e)

    if not USE_AGENT:
        print("▶ USE_AGENT=0 — deterministic pipeline only.")
    elif run_agent_system is None:
        print("▶ USE_AGENT=1 but agent system module not available — will attempt pipeline fallback.")

    # choose execution path
    if USE_AGENT and run_agent_system is not None:
        try:
            print("🔷 Running Agent system (attempt)...")
            run_agent_system(use_mock=USE_MOCK, outputs_dir="outputs")
            return
        except Exception as e:
            print("⚠️ Agent execution failed, falling back to pipeline. Exception:")
            traceback.print_exc()

    if run_pipeline is not None:
        print("▶ Running deterministic pipeline...")
        run_pipeline(use_mock=USE_MOCK, outputs_dir="outputs")
    else:
        print("❌ No pipeline available. Ensure file src/agents/langchain_pipeline.py exists and defines run_pipeline().")

if __name__ == "__main__":
    main()
