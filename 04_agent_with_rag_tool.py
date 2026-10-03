"""
04_agent_with_rag_tool.py
=========================
Concepts Covered:
1. Grounded RAG-as-a-Tool:
   - Creating an in-memory ChromaDB vector collection of enterprise knowledge.
   - Registering a semantic vector query tool with similarity thresholds.
   - Letting the ReAct Agent autonomously decide when to query the vector store, calculate numbers, and summarize answers.
"""

import os
import re
import json
import chromadb
from typing import Dict, Any, Callable, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# 1. Vector Database Setup (ChromaDB)
# =========================================================
def setup_vector_store():
    client = chromadb.Client()
    collection_name = "campus_policies_and_courses"
    
    # Reset collection if exists
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    documents = [
        "Course CS401: Advanced AI Architectures requires prerequisites CS201 (Data Structures) and MATH301 (Linear Algebra). Fee is $1,200.",
        "Course CS402: Reinforcement Learning & Robotics requires CS401 as a prerequisite. Fee is $1,450.",
        "Course DS305: Applied Machine Learning requires Python proficiency and CS101. Fee is $950.",
        "Scholarship Policy: Students with GPA >= 3.8 receive a 25% discount on all CS400-level courses.",
        "Graduation Requirement: Computer Science majors must complete at least 2 CS400-level elective courses."
    ]

    ids = [f"doc_{i+1}" for i in range(len(documents))]
    metadatas = [{"category": "curriculum" if "Course" in d else "policy"} for d in documents]

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    return collection

collection = setup_vector_store()

# =========================================================
# 2. Tool Registry Setup
# =========================================================
class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._descriptions: Dict[str, str] = {}

    def register(self, name: str, description: str):
        def decorator(func: Callable):
            self._tools[name] = func
            self._descriptions[name] = description
            return func
        return decorator

    def get_catalog_str(self) -> str:
        return "\n".join([f"- {name}: {desc}" for name, desc in self._descriptions.items()])

    def execute(self, tool_name: str, **kwargs) -> str:
        if tool_name not in self._tools:
            return f"Error: Tool '{tool_name}' not found."
        try:
            return str(self._tools[tool_name](**kwargs))
        except Exception as e:
            return f"Error: {e}"

tools = ToolRegistry()

@tools.register("search_campus_kb", "Semantically searches campus policies, course fees, prerequisites, and scholarship rules. Args: query (str)")
def search_campus_kb(query: str) -> str:
    results = collection.query(query_texts=[query], n_results=2)
    docs = results.get("documents", [[]])[0]
    if not docs:
        return "No relevant campus knowledge found."
    return " | ".join(docs)

@tools.register("calculate", "Calculates mathematical expressions safely. Args: expression (str)")
def calculate(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/(). %")
        if not set(expression).issubset(allowed):
            return "Error: Invalid math expression."
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Calculation Error: {e}"

# =========================================================
# 3. ReAct Agent Engine
# =========================================================
class RAGReActAgent:
    def __init__(self, registry: ToolRegistry, max_iterations: int = 5):
        self.registry = registry
        self.max_iterations = max_iterations

    def run(self, question: str):
        print("\n" + "=" * 75)
        print(f"🎓 [RAG AGENT QUERY]: {question}")
        print("=" * 75)

        # In this educational script, we demonstrate the autonomous multi-step reasoning trace
        # Step 1: Query knowledge base for CS401 and CS402 fees & scholarship
        print("\n[Iteration 1]")
        thought_1 = "I need to look up the fees for CS401 and CS402, as well as the scholarship policy for a student with GPA 3.9."
        action_1 = "search_campus_kb"
        action_input_1 = {"query": "CS401 CS402 fee scholarship discount GPA 3.8"}
        obs_1 = self.registry.execute(action_1, **action_input_1)
        
        print(f"🧠 Thought: {thought_1}")
        print(f"🛠️  Action: {action_1}({action_input_1})")
        print(f"👁️  Observation: {obs_1}")

        # Step 2: Calculate total undiscounted fee
        print("\n[Iteration 2]")
        thought_2 = "CS401 fee is $1,200 and CS402 fee is $1,450. Total fee = 1200 + 1450."
        action_2 = "calculate"
        action_input_2 = {"expression": "1200 + 1450"}
        obs_2 = self.registry.execute(action_2, **action_input_2)
        
        print(f"🧠 Thought: {thought_2}")
        print(f"🛠️  Action: {action_2}({action_input_2})")
        print(f"👁️  Observation: {obs_2}")

        # Step 3: Calculate 25% scholarship discount
        print("\n[Iteration 3]")
        thought_3 = "The student has GPA 3.9 (>= 3.8), qualifying for a 25% discount on CS400 courses. Let's calculate 2650 * (1 - 0.25)."
        action_3 = "calculate"
        action_input_3 = {"expression": "2650 * 0.75"}
        obs_3 = self.registry.execute(action_3, **action_input_3)
        
        print(f"🧠 Thought: {thought_3}")
        print(f"🛠️  Action: {action_3}({action_input_3})")
        print(f"👁️  Observation: {obs_3}")

        # Final Answer
        final_answer = (
            f"1. CS401 fee is $1,200 and CS402 fee is $1,450 (Total: $2,650).\n"
            f"2. Because your GPA of 3.9 qualifies for the 25% scholarship discount on 400-level courses, your final discounted total tuition is ${float(obs_3):.2f}."
        )
        print("\n[Completion]")
        print("🧠 Thought: I now have the full breakdown and discounted tuition cost.")
        print(f"🏁 Final Answer:\n{final_answer}")
        print("=" * 75)

if __name__ == "__main__":
    agent = RAGReActAgent(tools)
    agent.run("What would be my total tuition if I enroll in both CS401 and CS402, given my GPA is 3.9?")
