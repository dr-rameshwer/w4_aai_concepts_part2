# Week 4 (Part 2): Advanced AI Concepts & Pure Python ReAct Agents

This repository provides hands-on implementations and deep dives into **Pure Python ReAct Agents**, **Dynamic Tool Registries**, **Error Boundary Handling**, and **RAG-as-a-Tool Integration** (without n8n workflows).

---

## 📂 Modules & Exercises

### 1. `01_react_loop_anatomy.py`
- **Focus**: The core mental model of ReAct (Reasoning + Acting).
- **Phases**:
  - `Thought`: Self-reflection and goal formulation.
  - `Action`: Selecting external tools and extracting parameter inputs.
  - `Observation`: Executing the tool in the host environment and returning output.
  - `Final Answer`: Synthesizing collected evidence into a coherent response.

### 2. `02_dynamic_tool_registry.py`
- **Focus**: Tool Registry Architecture.
- **Features**:
  - Decorator-based registration (`@registry.register_tool`).
  - Automatic type inspection via Python's `inspect` module.
  - LLM system prompt catalog generator.
  - Error boundaries preventing script termination on invalid parameters or missing tools.

### 3. `03_pure_python_react_agent.py`
- **Focus**: Full Autonomous ReAct Engine.
- **Features**:
  - System prompt enforcing structured ReAct steps.
  - Dynamic scratchpad maintaining multi-turn context.
  - Loop execution with `max_iterations` safeguard.
  - Compatible with `google.genai` (Gemini) and fallback heuristic mode.

### 4. `04_agent_with_rag_tool.py`
- **Focus**: Vector Search & Grounded RAG as an Agent Tool.
- **Features**:
  - In-memory ChromaDB vector collection.
  - Multi-hop reasoning combining semantic retrieval + exact arithmetic calculation.

---

## 🚀 How to Run

Run any of the modules using Python:

```bash
python3 01_react_loop_anatomy.py
python3 02_dynamic_tool_registry.py
python3 03_pure_python_react_agent.py
python3 04_agent_with_rag_tool.py
```
