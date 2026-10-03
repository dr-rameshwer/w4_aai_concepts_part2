"""
01_react_loop_anatomy.py
========================
Concepts Covered:
1. The ReAct (Reasoning + Acting) Paradigm:
   - Thought: Agent reasons about what it needs to achieve and what to do next.
   - Action: Agent selects a tool and provides input parameters.
   - Observation: Execution output from the tool environment.
   - Final Answer: Concluding response when no further actions are required.
2. Step-by-step trace simulation to understand the mental model.
"""

import os
import json
import re
from typing import Dict, Any, Callable

# ---------------------------------------------------------
# Mock Environment / External Tools
# ---------------------------------------------------------
def search_database(query: str) -> str:
    """Mock search tool simulating external data lookup."""
    knowledge_base = {
        "apple revenue 2024": "Apple reported $391.04 billion in total net sales for fiscal year 2024.",
        "microsoft revenue 2024": "Microsoft reported $245.1 billion in revenue for fiscal year 2024.",
        "weather tokyo": "The current weather in Tokyo is 18°C, Partly Cloudy.",
        "currency usd inr": "1 USD is approximately 84.0 INR."
    }
    for key, val in knowledge_base.items():
        if all(word in query.lower() for word in key.split()):
            return val
    return f"No direct match found for '{query}'. Try a different keyword."

def calculate(expression: str) -> str:
    """Safe evaluation calculator tool."""
    try:
        # Allow only safe mathematical operations
        allowed_chars = set("0123456789+-*/(). %")
        if not set(expression).issubset(allowed_chars):
            return "Error: Invalid characters in arithmetic expression."
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Calculation Error: {e}"

# Tool Dispatch Table
AVAILABLE_TOOLS: Dict[str, Callable[[str], str]] = {
    "search_database": search_database,
    "calculate": calculate
}

# ---------------------------------------------------------
# Manual ReAct Step-by-Step Walkthrough
# ---------------------------------------------------------
def demonstrate_react_trace():
    print("=" * 70)
    print("DEMO: Anatomy of a ReAct Loop")
    print("=" * 70)
    
    user_prompt = "What is the difference between Apple's and Microsoft's FY2024 revenues?"
    print(f"\n[USER QUESTION]: {user_prompt}\n")

    # Step 1: Agent decides to look up Apple's revenue
    step1_thought = "I need to find Apple's fiscal year 2024 revenue first."
    step1_action = "search_database"
    step1_action_input = "apple revenue 2024"
    
    print("--> [STEP 1]")
    print(f"    Thought: {step1_thought}")
    print(f"    Action: {step1_action}({step1_action_input})")
    
    # Execute Action 1
    step1_observation = AVAILABLE_TOOLS[step1_action](step1_action_input)
    print(f"    Observation: {step1_observation}\n")

    # Step 2: Agent decides to look up Microsoft's revenue
    step2_thought = "Now I need to find Microsoft's fiscal year 2024 revenue."
    step2_action = "search_database"
    step2_action_input = "microsoft revenue 2024"
    
    print("--> [STEP 2]")
    print(f"    Thought: {step2_thought}")
    print(f"    Action: {step2_action}({step2_action_input})")
    
    # Execute Action 2
    step2_observation = AVAILABLE_TOOLS[step2_action](step2_action_input)
    print(f"    Observation: {step2_observation}\n")

    # Step 3: Agent computes the difference
    step3_thought = "Apple revenue is $391.04B and Microsoft revenue is $245.1B. I will calculate 391.04 - 245.1."
    step3_action = "calculate"
    step3_action_input = "391.04 - 245.1"
    
    print("--> [STEP 3]")
    print(f"    Thought: {step3_thought}")
    print(f"    Action: {step3_action}({step3_action_input})")
    
    # Execute Action 3
    step3_observation = AVAILABLE_TOOLS[step3_action](step3_action_input)
    print(f"    Observation: {step3_observation}\n")

    # Step 4: Final Answer
    step4_thought = "I now have all the data required to answer the user's question."
    final_answer = f"In FY2024, Apple earned $391.04 billion and Microsoft earned $245.10 billion. The difference is ${float(step3_observation):.2f} billion."
    
    print("--> [COMPLETION]")
    print(f"    Thought: {step4_thought}")
    print(f"    Final Answer: {final_answer}\n")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_react_trace()
