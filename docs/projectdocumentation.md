Project Documentation
Multi-Agent Content Generation System (LangChain + LangGraph)
Applied AI Engineer Challenge — Final Submission
1. Introduction

This project implements an AI-driven multi-agent content generation system designed to autonomously produce structured product content for e-commerce platforms.
Unlike a static script, this system uses LangChain’s agent framework (with optional LangGraph orchestration) to create:

Product Page

FAQ Section

Comparison Page

Normalized JSON Output

The pipeline ensures modular, reusable, schema-controlled generation via specialized agents and tool-based reasoning.

This redesigned version fulfills the assignment requirement of using a mature agentic framework instead of custom orchestration logic.

2. Objectives

The system is designed to:

✔ Normalize noisy product input into a strict schema
✔ Generate FAQs using a Question Generation agent
✔ Answer FAQs using an Answering agent
✔ Assemble all outputs using an Assembler agent
✔ Operate under LangChain agent/tool orchestration
✔ Produce final outputs as clean, valid JSON files
✔ Support multiple LLM backends (Mock, HF local, or cloud models)
3. System Architecture
3.1 High-Level Agentic Architecture
                    ┌────────────────────┐
                    │  Input Product JSON │
                    └─────────┬──────────┘
                              │
        ┌─────────────────────┴──────────────────────────┐
        │                                                │
┌───────────────┐  ┌─────────────────┐   ┌──────────────────────────┐
│ Parser Agent  │  │ Question Agent  │   │  Answer Agent            │
│ (JSON Normal) │  │ (QGen)          │   │  (FAQ Answering)         │
└───────┬───────┘  └────────┬────────┘   └──────────┬───────────────┘
        │                   │                       │
        └───────────────┬──┴───────┬───────────────┘
                        │          │
              ┌─────────▼──────────▼─────────┐
              │     Assembler Agent          │
              │ (Product Page + FAQs + Comp) │
              └───────────────┬──────────────┘
                              │
                     ┌────────▼────────┐
                     │ JSON File Writer│
                     └─────────────────┘

3.2 Components
1. Parser Agent

Validates and normalizes product JSON.

Ensures mandatory keys exist:

product_name

concentration

skin_type

key_ingredients

benefits

how_to_use

side_effects

price

Tools used: LangChain LLM + custom validation function.

2. Question Generation Agent (QGen Agent)

Generates 15+ diverse FAQs based on the normalized product JSON.

Output categories include:

Informational

Usage

Safety

Purchase

Comparison

Uses LangChain’s prompt templates + LLM.

3. FAQ Answering Agent

Answers each generated FAQ.

Must use only product JSON input (no hallucinations).

Produces:

{
  "question": "...",
  "answer": "...",
  "category": "..."
}

4. Assembler Agent

Combines:

Product Page Data

FAQ list

Competitor comparison

Uses Jinja templates for consistent structure.

Ensures output matches strict JSON schemas.

5. File Writer Tool

Saves JSON to:

/outputs/product_page.json  
/outputs/faq.json  
/outputs/comparison_page.json  

4. Technologies Used
Component	Technology
Agent Framework	LangChain
Optional Orchestration	LangGraph
LLM	HuggingFace Local Models (e.g., flan-t5-small, distilgpt2)
Prompts	LangChain PromptTemplate
Templates	Jinja2
Storage	JSON Output
Environment	Python 3.10+
5. Folder Structure
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

6. Execution Flow

1️⃣ Load Input JSON → product_input.json

2️⃣ Parser Agent
Normalizes data → returns clean schema.

3️⃣ QGen Agent
Creates 15+ FAQs.

4️⃣ Answer Agent
Generates accurate answers using product facts.

5️⃣ Assembler Agent
Builds product page + comparison page.

6️⃣ Writer Tool
Saves final JSON files.

7. Running the Project
Mock Mode (for testing)
$env:USE_MOCK="1"
python -m src.main

Local LLM Mode
$env:USE_MOCK="0"
$env:USE_LOCAL_LLM="1"
$env:LOCAL_MODEL_NAME="google/flan-t5-small"
$env:HF_PIPELINE_TASK="text2text-generation"
python -m src.main


Outputs appear in the /outputs folder.

8. Example Outputs
Product Page
{
  "title": "GlowBoost Vitamin C Serum",
  "concentration": "10% Vitamin C",
  "skin_types": ["Oily", "Combination"],
  "key_ingredients": ["Vitamin C", "Hyaluronic Acid"],
  ...
}

FAQ
[
  {
    "question": "What is this serum used for?",
    "answer": "It helps brighten skin and fade dark spots.",
    "category": "Informational"
  },
  ...
]

Comparison Page
{
  "base": {"name": "GlowBoost Vitamin C Serum", "price": "₹699"},
  "competitor": {"name": "Fictional B", "price": "₹1,199"}
}

9. Why This System Meets Assignment Requirements
✔ Uses LangChain agents, not custom orchestration
✔ Multiple specialized agents with clear responsibilities
✔ Prompt-driven, model-based generation (no hardcoded logic)
✔ JSON schema enforcement
✔ Separation of concerns via tools + prompts
✔ Clean, production-ready architecture
✔ Fully reproducible pipeline with mock mode
10. Limitations & Future Work

Add evaluator agent to rate answer quality

Expand to multilingual content

Integrate vector search for long product descriptions

Containerize using Docker

Add web UI for input/upload

11. Conclusion

This final submission demonstrates a fully agentic system built on LangChain, featuring:

parser agent

question generator agent

FAQ answering agent

assembler agent

file-writing tools

structured output pipeline

The project satisfies the AI Engineering Challenge expectations with a clean, framework-driven architecture and reusable components.

End of Documentation