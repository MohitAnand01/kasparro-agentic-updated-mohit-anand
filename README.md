# Kasparro Multi-Agent Content System (LangChain Version)
**Author:** Mohit Anand

## Overview
This repository implements a LangChain-based multi-agent pipeline that converts a small product dataset (GlowBoost Vitamin C Serum) into three structured JSON pages:
- product_page.json
- faq.json
- comparison_page.json

## Run locally
1. Create a venv (optional) and activate:
   - Windows:
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

2. Install:
   ```bash
   pip install -r requirements.txt
