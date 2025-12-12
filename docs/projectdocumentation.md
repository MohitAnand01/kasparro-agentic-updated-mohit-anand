project:
  name: "Multi-Agent Content Generation System"
  framework: "LangChain + LangGraph"
  challenge: "Applied AI Engineer Challenge — Final Submission"

introduction:
  description: >
    An AI-driven multi-agent system that autonomously generates structured
    product content for e-commerce platforms. Uses LangChain’s agent framework
    with optional LangGraph orchestration. Produces product pages, FAQ
    sections, comparison pages, and normalized JSON outputs.

objectives:
  - "Normalize noisy product input into a strict schema"
  - "Generate FAQs using a Question Generation agent"
  - "Answer FAQs using an Answering agent"
  - "Assemble all outputs using an Assembler agent"
  - "Operate under LangChain agent/tool orchestration"
  - "Produce clean, valid JSON files"
  - "Support multiple LLM backends (Mock, HF local, or cloud models)"

architecture:
  high_level_diagram: |
    ┌────────────────────┐
    │  Input Product JSON │
    └─────────┬──────────┘
              │
    ┌──────────────────────────────────────────────┐
    │                                              │
    │  ┌───────────────┐  ┌─────────────────┐      ┌──────────────────────────┐
    │  │ Parser Agent  │  │ Question Agent  │      │  Answer Agent            │
    │  │ (JSON Normal) │  │ (QGen)          │      │  (FAQ Answering)         │
    │  └───────┬───────┘  └────────┬────────┘      └──────────┬───────────────┘
    │          │                    │                           │
    │          └───────────────┬────┴───────┬──────────────────┘
    │                          │            │
    │                ┌─────────▼────────────▼─────────┐
    │                │     Assembler Agent            │
    │                │ (Product Page + FAQs + Comp)   │
    │                └───────────────┬────────────────┘
    │                                │
    │                       ┌────────▼────────┐
    │                       │ JSON File Writer│
    │                       └─────────────────┘
    └────────────────────────────────────────────────┘

components:
  parser_agent:
    role: "Validates and normalizes product JSON"
    ensures_fields:
      - product_name
      - concentration
      - skin_type
      - key_ingredients
      - benefits
      - how_to_use
      - side_effects
      - price
    tools_used:
      - "LangChain LLM"
      - "Custom validation functions"

  question_generation_agent:
    role: "Generates 15+ FAQs using LangChain prompts"
    categories:
      - Informational
      - Usage
      - Safety
      - Purchase
      - Comparison

  faq_answering_agent:
    role: "Answers FAQs using only provided product facts"
    output_format:
      question: "string"
      answer: "string"
      category: "string"

  assembler_agent:
    role: "Combines product page, FAQ list, and comparison page"
    technology: "Jinja2 templates"
    guarantee: "Strict JSON schema adherence"

  file_writer:
    outputs:
      - outputs/product_page.json
      - outputs/faq.json
      - outputs/comparison_page.json

technologies:
  agent_framework: "LangChain"
  orchestration: "LangGraph (optional)"
  llm_backends:
    - "HuggingFace Local Models (flan-t5-small, distilgpt2)"
  prompting: "LangChain PromptTemplate"
  templating: "Jinja2"
  storage: "JSON"
  environment: "Python 3.10+"

folder_structure:
  src:
    agents:
      - langchain_agent_system.py
      - langchain_pipeline.py
    tools:
      - llm_tools.py
      - file_tools.py
    prompts:
      - parser_prompt.txt
      - qgen_prompt.txt
      - planner_prompt.txt
      - assembler_prompt.txt
    templates:
      - product_template.j2
    data:
      - product_input.json
    main: main.py
    init: "__init__.py"
  outputs:
    - product_page.json
    - faq.json
    - comparison_page.json

execution_flow:
  - step: "Load Input JSON"
    file: "product_input.json"
  - step: "Parser Agent"
    result: "Normalized schema"
  - step: "Question Generation Agent"
    result: "15+ FAQs"
  - step: "FAQ Answering Agent"
    result: "Fact-based answers"
  - step: "Assembler Agent"
    result: "Product page + FAQ page + comparison page"
  - step: "File Writer"
    result: "Final JSON files saved"

running:
  mock_mode:
    commands:
      - '$env:USE_MOCK="1"'
      - "python -m src.main"
  local_llm_mode:
    commands:
      - '$env:USE_MOCK="0"'
      - '$env:USE_LOCAL_LLM="1"'
      - '$env:LOCAL_MODEL_NAME="google/flan-t5-small"'
      - '$env:HF_PIPELINE_TASK="text2text-generation"'
      - "python -m src.main"
  output_location: "/outputs"

example_outputs:
  product_page:
    title: "GlowBoost Vitamin C Serum"
    concentration: "10% Vitamin C"
    skin_types: ["Oily", "Combination"]
    key_ingredients:
      - "Vitamin C"
      - "Hyaluronic Acid"
  faq:
    - question: "What is this serum used for?"
      answer: "It helps brighten skin and fade dark spots."
      category: "Informational"
  comparison_page:
    base:
      name: "GlowBoost Vitamin C Serum"
      price: "₹699"
    competitor:
      name: "Fictional B"
      price: "₹1,199"

requirements_met:
  - "Uses LangChain agents (not custom code)"
  - "Specialized agents with clear roles"
  - "Prompt-driven, LLM-based reasoning"
  - "Strict JSON schema enforcement"
  - "Modular & maintainable architecture"
  - "Supports mock + local models"
  - "Fully reproducible pipeline"

limitations_future_work:
  - "Add evaluator agent for answer quality"
  - "Support multilingual content"
  - "Integrate vector search for long inputs"
  - "Dockerize entire system"
  - "Add web-based UI for input/upload"

conclusion: >
  A complete agentic system built using LangChain, featuring a parser agent,
  question generation agent, FAQ answering agent, assembler agent, and file
  writing tools. Fully satisfies challenge requirements and provides a scalable,
  production-ready pipeline for automated e-commerce content creation.
