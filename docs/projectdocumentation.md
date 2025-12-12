📘 Multi-Agent Content Generation System (LangChain + LangGraph)
Applied AI Engineer Challenge — Final Submission
Author: Mohit Anand
1. Problem Statement

Design and implement a modular agentic automation system that transforms a small product dataset into:

FAQ Page (15+ Q&As)

Product Description Page

Comparison Page

Normalized JSON Outputs

The system must:

Use multiple independent agents

Demonstrate an orchestration flow / automation graph

Contain reusable logic blocks

Use a custom template engine

Output clean JSON files

Use only the provided dataset, with no external facts or internet lookup

The goal is to evaluate:

System design

Agent architecture

Orchestration

Reusability and reliability

2. Solution Overview

The solution is built as a four-agent modular system, orchestrated by a LangChain-based workflow.
Each agent performs one atomic responsibility, ensuring deterministic and clean outputs.

🌟 The Four Agents

Parser Agent
Normalizes raw JSON → Validated structured schema.

Question Generation Agent
Produces 15+ categorized FAQs using controlled prompt logic.

FAQ Answering Agent
Answers questions strictly using product facts (no hallucinations).

Assembler Agent
Uses templates + logic blocks to build:

Product Page JSON

FAQ Page JSON

Comparison Page JSON

🔧 Supporting Modules

Template Engine (Jinja2)

Reusable Logic Blocks

LangChain Tooling

Writer Tool for File Output

3. Scopes & Assumptions
✔️ In Scope

Parsing and validating the GlowBoost dataset

Structured question generation

Fact-based FAQ construction

Template-driven page creation

JSON-only output

Multi-agent workflow coordination

❌ Out of Scope

External APIs / online data

GPT or creative content writing

UI, frontend, or web server

Dataset modification

Assumptions

Input always follows expected schema

System must run without internet

Architecture should remain extensible for future model integrations

4. System Design

The system follows a 4-stage pipeline, where each stage produces intermediate structured data stored in a shared PageContext.

4.1 High-Level System Architecture Diagram

This diagram shows how modules, agents, tools, schema validators, and templates integrate:

flowchart LR

    subgraph INPUT[Input Layer]
        A[Raw Product JSON]
    end

    subgraph AGENTS[Agent Layer]
        P[Parser Agent\nNormalize + Validate Schema]
        QG[Question Generation Agent\n15+ Categorized Questions]
        ANS[FAQ Answering Agent\nFact-Based Answers]
        ASM[Assembler Agent\nTemplates + Logic Blocks]
    end

    subgraph LOGIC[Supporting Logic]
        LB[Reusable Logic Blocks\nUsage Rules · Safety · Benefits]
        TMP[Template Engine\n(Jinja2)]
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

4.2 Agent Workflow Diagram

This diagram visualizes the core agentic content-generation pipeline:

flowchart TD

    A[Raw Product JSON] --> B[Parser Agent<br/>Normalize & Validate Schema]
    B --> C[Question Generation Agent<br/>Generate 15+ Categorized FAQs]
    C --> D[FAQ Answering Agent<br/>Answer FAQs Using Product Facts Only]
    D --> E[Assembler Agent<br/>Build Product · FAQ · Comparison Pages]

    E --> F1[product_page.json]
    E --> F2[faq.json]
    E --> F3[comparison_page.json]

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

Load Product JSON

Parser Agent normalizes and validates JSON

QGen Agent produces 15+ categorized FAQs

Answer Agent answers FAQs using product facts only

Assembler Agent builds structured pages via templates

Writer Tool outputs final JSON files

7. Tech Stack
Component	Technology
Agent Framework	LangChain
Optional Orchestration	LangGraph
LLM Backend	HuggingFace Local Models
Prompts	LangChain PromptTemplate
Template Engine	Jinja2
Output Format	JSON
Language	Python 3.10+
8. Why This Meets All Challenge Requirements

✔ Multiple agents, each with clear responsibilities
✔ Controlled orchestration with LangChain
✔ Reusable logic blocks
✔ Custom templating engine
✔ Clean, deterministic JSON output
✔ Works fully offline
✔ Zero external knowledge
✔ Production-grade modularity

This project cleanly demonstrates professional-level agentic system design.

9. Conclusion

This system transforms raw product data into structured, machine-ready content through a robust, four-agent pipeline.
Using LangChain’s agent framework, reusable logic blocks, and a template-driven architecture, the system delivers:

Product Page

FAQ Page

Comparison Page

Clean, validated JSON outputs

It fully satisfies the Applied AI Engineer Challenge requirements with a scalable, maintainable, and production-ready architecture.
