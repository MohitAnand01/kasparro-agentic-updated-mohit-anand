📘 Multi-Agent Content Generation System
LangChain + LangGraph | Applied AI Engineer Challenge

Author: Mohit Anand

🚀 1. Problem Statement

Develop a modular agentic automation system capable of transforming a small product dataset into:

🟦 Product Description Page

🟩 FAQ Page (15+ structured questions & answers)

🟥 Comparison Page

🟨 Normalized JSON Outputs

✅ System Requirements

The system must:

Use multiple, independently functioning agents

Demonstrate a clear orchestration / automation flow

Utilize reusable logic blocks

Use a custom template engine

Produce machine-readable JSON outputs

Use only the provided dataset (no external facts or internet lookup)

🎯 Evaluation Focus

Architecture & system design

Multi-agent orchestration

Modularity & reusability

Structured deterministic output

🧠 2. Solution Overview

The solution is a four-agent deterministic architecture orchestrated via LangChain.

Each agent performs a single responsibility, enabling:
✔ Maintainability
✔ Determinism
✔ Testability
✔ Extensibility

🔧 2.1 The Four Core Agents
1️⃣ Parser Agent

Validates raw JSON

Ensures required fields exist

Converts dataset into a strict normalized schema

Initializes PageContext shared across agents

2️⃣ Question Generation Agent

Produces 15+ categorized FAQs

Uses prompt logic instead of AI creativity

Categories include:

Informational

Usage

Safety

Purchase

Comparison

3️⃣ FAQ Answering Agent

Answers each FAQ using only the provided product facts

Zero hallucination

Output schema:

{
  "question": "...",
  "answer": "...",
  "category": "..."
}

4️⃣ Page Assembler Agent

Uses templates + logic rules to generate:

Product Page JSON

FAQ JSON

Comparison Page JSON

Enforces strict schemas

Powered by Jinja2 template engine

📦 3. Scope & Assumptions
✔️ In Scope

Parsing & validating provided dataset

FAQ generation (≥ 15)

Fact-based FAQ answering

Template-driven page assembly

Strict JSON output

Offline execution

❌ Out of Scope

Internet access / external data sources

Creative rewriting or LLM hallucinations

UI / frontend

Dataset expansion

📌 Assumptions

Dataset always follows expected schema

System should remain modular for future upgrades

No external facts may be introduced

🏗️ 4. System Design

The system follows a four-stage agentic pipeline, each transforming the data before passing it forward. Outputs are stored in an evolving shared PageContext.

🖥️ 4.1 High-Level System Architecture
flowchart LR

    subgraph INPUT[Input Layer]
        A[Raw Product JSON]
    end

    subgraph AGENTS[Agent Layer]
        P[Parser Agent<br/>Normalize + Validate Schema]
        QG[Question Generation Agent<br/>15+ Categorized Questions]
        ANS[FAQ Answering Agent<br/>Fact-Based Answers]
        ASM[Assembler Agent<br/>Templates + Logic Blocks]
    end

    subgraph LOGIC[Supporting Logic]
        LB[Reusable Logic Blocks<br/>Usage · Safety · Benefits]
        TMP[Template Engine (Jinja2)]
        VAL[Schema Validation]
    end

    subgraph OUTPUT[Output Layer]
        OP1[product_page.json]
        OP2[faq.json]
        OP3[comparison_page.json]
    end

    A --> P --> QG --> ANS --> ASM
    LB --> ASM
    TMP --> ASM
    VAL --> P

    ASM --> OP1
    ASM --> OP2
    ASM --> OP3

🔄 4.2 Agent Workflow Pipeline
flowchart TD

    A[Raw Product JSON] --> B[Parser Agent<br/>Normalize & Validate Schema]
    B --> C[Question Generation Agent<br/>Generate 15+ Categorized FAQs]
    C --> D[FAQ Answering Agent<br/>Answer Using Product Facts Only]
    D --> E[Assembler Agent<br/>Build Product · FAQ · Comparison Pages]

    E --> F1[product_page.json]
    E --> F2[faq.json]
    E --> F3[comparison_page.json]

📁 5. Folder Structure (GitHub-Ready)
src/
 ├── agents/
 │    ├── langchain_agent_system.py
 │    └── langchain_pipeline.py
 ├── tools/
 │    ├── llm_tools.py
 │    └── file_tools.py
 ├── prompts/
 │    ├── parser_prompt.txt
 │    ├── qgen_prompt.txt
 │    ├── planner_prompt.txt
 │    └── assembler_prompt.txt
 ├── templates/
 │    └── product_template.j2
 ├── data/
 │    └── product_input.json
 ├── main.py
 └── __init__.py

outputs/
 ├── product_page.json
 ├── faq.json
 └── comparison_page.json

⚙️ 6. Execution Flow
Step 1 — Load Input JSON

Loads product_input.json.

Step 2 — Parser Agent

Validates → Normalizes → Creates internal schema.

Step 3 — Question Generation Agent

Produces 15+ structured questions.

Step 4 — FAQ Answering Agent

Answers questions using only product data.

Step 5 — Assembler Agent

Uses templates + logic blocks to create output pages.

Step 6 — File Writer Tool

Exports all three pages as JSON.

🧰 7. Tech Stack
Component	Technology
Agent Framework	LangChain
Optional Orchestration	LangGraph
LLM Backend	HuggingFace (flan-t5-small, distilgpt2)
Prompt Engine	LangChain PromptTemplate
Template Engine	Jinja2
Output Format	JSON
Language	Python 3.10+
🏆 8. Why This Solution Meets All Requirements

✔ Multi-agent architecture
✔ Framework-driven agent orchestration
✔ Reusable logic blocks
✔ Custom Jinja2 template engine
✔ Deterministic output (Mock or Local LLM mode)
✔ Clean JSON schema enforcement
✔ Offline-friendly
✔ Maintainable folder structure
✔ Zero hallucinations (facts only from product JSON)

🎯 9. Conclusion

This project demonstrates a production-ready agentic automation pipeline powered by LangChain.
Through strict schema enforcement, modular agent design, and template-driven output generation, the system reliably produces:

Product Description Page

FAQ Page (15+ items)

Comparison Page

Structured JSON outputs

The architecture is:

Scalable

Maintainable

Deterministic

Fully challenge compliant

If you'd like:
✅ A PDF-ready version
✅ A GitHub Pages documentation version
✅ A compressed 1-page executive summary

Just tell me!
