# src/tools/file_tools.py
from pathlib import Path
import json, os, sys

def ensure_outputs_dir(path: str = "outputs"):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

def load_product_json_text(path: str = "src/data/product_input.json") -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")

def write_json_file(fname: str, data, outputs_dir: str = "outputs"):
    ensure_outputs_dir(outputs_dir)
    outp = Path(outputs_dir) / fname
    outp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Wrote {outp}")
