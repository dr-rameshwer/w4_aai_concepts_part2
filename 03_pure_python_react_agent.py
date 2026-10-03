"""
03_pure_python_react_agent.py
=============================
Concepts Covered:
1. Pure Python Autonomous ReAct Agent:
   - System Prompting with strict ReAct instructions.
   - Parsing engine: Extracting Thought, Action, and Action Input.
   - Scratchpad & Context Accumulation: Appending Observations back to the prompt history.
   - Loop execution with Max Iterations safeguard to prevent infinite loops.
   - Tool execution error handling and self-correction in subsequent iterations.
   - Real LLM invocation (Gemini / OpenAI) with graceful fallback simulation.
"""

import os
import re
import json
from typing import Dict, Any, Callable, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# 1. Dynamic Tool Registry
# =========================================================
class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._descriptions: Dict[str, str] = {}

    def register(self, name: Optional[str] = None, description: str = ""):
        def decorator(func: Callable):
            tool_name = name or func.__name__
            self._tools[tool_name] = func
            self._descriptions[tool_name] = description or (func.__doc__ or "").strip()
            return func
        return decorator

    def get_catalog_str(self) -> str:
        lines = []
        for name, desc in self._descriptions.items():
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def execute(self, tool_name: str, **kwargs) -> str:
        if tool_name not in self._tools:
            return f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(self._tools.keys())}"
        try:
            return str(self._tools[tool_name](**kwargs))
        except Exception as e:
            return f"Error executing '{tool_name}': {str(e)}"

# Instantiate tools
registry = ToolRegistry()

@registry.register(name="search_wiki", description="Lookup factual knowledge, events, and metrics. Args: query (str)")
def search_wiki(query: str) -> str:
    db = {
        "jupiter moons": "Jupiter has 95 officially recognized moons, including Ganymede, Callisto, Io, and Europa.",
        "ganymede size": "Ganymede has a diameter of 5,268 kilometers, making it the largest moon in the Solar System.",
        "titan size": "Titan has a diameter of 5,149.5 kilometers, making it the second largest moon.",
        "earth moon size": "Earth's Moon has a diameter of 3,474 kilometers.",
        "mars moons": "Mars has two small moons: Phobos and Deimos."
    }
    for k, v in db.items():
        if all(w in query.lower() for w in k.split()):
            return v
    return f"No direct entry found for '{query}'."

@registry.register(name="calculator", description="Perform exact arithmetic calculations. Args: expression (str)")
def calculator(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/(). %")
        if not set(expression).issubset(allowed):
            return "Error: Unsupported characters in math expression."
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Calculation Error: {e}"


# =========================================================
# 2. LLM Client Wrapper (Gemini 2.5 / Gemini-Flash / Fallback)
# =========================================================
def call_llm(prompt: str, stop_sequences: list = None) -> str:
    """Calls Gemini or falls back to standard LLM response."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
            )
            if response.text:
                return response.text
        except Exception as e:
            # Try legacy google.generativeai if available
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=api_key)
                model = legacy_genai.GenerativeModel("gemini-1.5-flash")
                resp = model.generate_content(prompt)
                if resp.text:
                    return resp.text
            except Exception:
                pass

    # Fallback heuristic engine if no API key is available
    if "ganymede" in prompt.lower() and "titan" in prompt.lower():
        if "Observation: Jupiter has 95" not in prompt and "Observation: Ganymede" not in prompt:
            return "Thought: I need to find the diameter of Ganymede.\nAction: search_wiki\nAction Input: {\"query\": \"ganymede size\"}"
        elif "Observation: Ganymede has a diameter of 5,268" in prompt and "Observation: Titan" not in prompt:
            return "Thought: Now I need to find the diameter of Titan.\nAction: search_wiki\nAction Input: {\"query\": \"titan size\"}"
        elif "Observation: Titan has a diameter of 5,149.5" in prompt and "5268 - 5149.5" not in prompt:
            return "Thought: I need to calculate the diameter difference between Ganymede and Titan.\nAction: calculator\nAction Input: {\"expression\": \"5268 - 5149.5\"}"
        else:
            return "Thought: I now have the calculated difference.\nFinal Answer: Ganymede (5,268 km) is larger than Titan (5,149.5 km) by 118.5 kilometers."
    
    return "Thought: I do not have enough context.\nFinal Answer: Unable to resolve without external API keys."


# =========================================================
# 3. Pure Python ReAct Agent
# =========================================================
class PurePythonReActAgent:
    def __init__(self, tool_registry: ToolRegistry, max_iterations: int = 6):
        self.tools = tool_registry
        self.max_iterations = max_iterations

    def _build_system_prompt(self, user_question: str) -> str:
        return f"""You are a helpful and precise reasoning AI assistant equipped with external tools.
Answer the user's question using the ReAct (Reasoning + Acting) loop.

AVAILABLE TOOLS:
{self.tools.get_catalog_str()}

INSTRUCTIONS:
You must strictly follow this format for each step:
Thought: Describe your reasoning about what you need to do next.
Action: tool_name
Action Input: {{"param_name": "param_value"}}

When you have the final answer, format it strictly as:
Thought: I have gathered all necessary information.
Final Answer: [Your complete, definitive answer here]

Question: {user_question}
"""

    def _parse_response(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]], Optional[str]]:
        """Extracts Thought, Action, Action Input, or Final Answer from LLM text."""
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", text, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else None

        final_answer_match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if final_answer_match:
            return thought, None, None, final_answer_match.group(1).strip()

        action_match = re.search(r"Action:\s*([a-zA-Z0-9_-]+)", text, re.IGNORECASE)
        action = action_match.group(1).strip() if action_match else None

        action_input = {}
        input_match = re.search(r"Action Input:\s*(\{.*?\}|.*)", text, re.DOTALL | re.IGNORECASE)
        if input_match:
            raw_input = input_match.group(1).strip()
            # Try parsing JSON
            try:
                action_input = json.loads(raw_input)
            except Exception:
                # Fallback: single string argument matching
                if action == "search_wiki":
                    action_input = {"query": raw_input.strip('"\'')}
                elif action == "calculator":
                    action_input = {"expression": raw_input.strip('"\'')}
                else:
                    action_input = {"query": raw_input}

        return thought, action, action_input, None

    def run(self, user_question: str) -> str:
        print("\n" + "=" * 75)
        print(f"🚀 [AGENT START] Question: {user_question}")
        print("=" * 75)

        prompt = self._build_system_prompt(user_question)
        scratchpad = ""

        for iteration in range(1, self.max_iterations + 1):
            current_prompt = prompt + scratchpad + "\n"
            
            # 1. Call LLM
            llm_output = call_llm(current_prompt)
            
            # 2. Parse output
            thought, action, action_input, final_answer = self._parse_response(llm_output)

            print(f"\n--- [Iteration {iteration}/{self.max_iterations}] ---")
            if thought:
                print(f"🧠 Thought: {thought}")

            # Check if finished
            if final_answer:
                print(f"🏁 Final Answer:\n{final_answer}")
                print("=" * 75)
                return final_answer

            if not action:
                print("⚠️ Warning: No valid action or final answer parsed. Retrying with error guidance.")
                scratchpad += f"\n{llm_output}\nObservation: Error: You must output either 'Action: tool_name' or 'Final Answer: ...'."
                continue

            print(f"🛠️  Action: {action}({action_input})")

            # 3. Execute Action
            observation = self.tools.execute(action, **action_input)
            print(f"👁️  Observation: {observation}")

            # 4. Append to Scratchpad
            step_record = f"\nThought: {thought}\nAction: {action}\nAction Input: {json.dumps(action_input)}\nObservation: {observation}"
            scratchpad += step_record

        return "Max iterations reached without achieving a final answer."


# =========================================================
# Run Agent Demonstration
# =========================================================
if __name__ == "__main__":
    agent = PurePythonReActAgent(tool_registry=registry)
    
    question = "How much larger in diameter is Ganymede compared to Titan?"
    result = agent.run(question)
