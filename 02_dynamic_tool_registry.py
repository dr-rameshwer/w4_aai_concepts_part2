"""
02_dynamic_tool_registry.py
===========================
Concepts Covered:
1. Dynamic Tool Registry Pattern:
   - Decorator-based tool registration (`@registry.register_tool`).
   - Automatic parameter and docstring extraction.
   - Dynamic tool schema generation for LLM system prompts.
   - Robust execution with error boundaries and argument coercion.
"""

import inspect
import json
from typing import Callable, Dict, Any, Optional

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, name: Optional[str] = None, description: Optional[str] = None):
        """Decorator to register a Python function as an agent tool."""
        def decorator(func: Callable):
            tool_name = name or func.__name__
            tool_doc = description or (func.__doc__ or "").strip()
            
            # Inspect parameters and type hints
            sig = inspect.signature(func)
            parameters = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                param_type = "string"
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == dict:
                    param_type = "object"
                elif param.annotation == list:
                    param_type = "array"

                parameters[param_name] = {
                    "type": param_type,
                    "default": None if param.default == inspect.Parameter.empty else param.default
                }
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            self._tools[tool_name] = func
            self._schemas[tool_name] = {
                "name": tool_name,
                "description": tool_doc,
                "parameters": parameters,
                "required": required
            }
            return func
        return decorator

    def get_tool_catalog_prompt(self) -> str:
        """Formats all registered tools into a prompt-friendly format for the LLM."""
        lines = []
        for name, schema in self._schemas.items():
            params = ", ".join([f"{k}: {v['type']}" for k, v in schema["parameters"].items()])
            lines.append(f"- {name}({params}): {schema['description']}")
        return "\n".join(lines)

    def execute(self, tool_name: str, **kwargs) -> str:
        """Executes a registered tool with safe error handling."""
        if tool_name not in self._tools:
            available = ", ".join(self._tools.keys())
            return f"Error: Tool '{tool_name}' does not exist. Available tools are: {available}"

        func = self._tools[tool_name]
        try:
            result = func(**kwargs)
            return str(result)
        except TypeError as e:
            return f"Error executing tool '{tool_name}': Invalid arguments provided ({e}). Expected parameters: {list(self._schemas[tool_name]['parameters'].keys())}"
        except Exception as e:
            return f"Error during tool '{tool_name}' execution: {str(e)}"


# ---------------------------------------------------------
# Define Tools Using the Registry
# ---------------------------------------------------------
registry = ToolRegistry()

@registry.register_tool(description="Fetches current stock price for a company ticker (e.g. AAPL, MSFT, GOOGL).")
def get_stock_price(ticker: str) -> str:
    ticker = ticker.upper().strip()
    mock_prices = {
        "AAPL": "$227.50",
        "MSFT": "$430.15",
        "GOOGL": "$182.30",
        "NVDA": "$125.80"
    }
    return mock_prices.get(ticker, f"Ticker '{ticker}' not found in mock market exchange.")

@registry.register_tool(description="Evaluates a mathematical expression safely.")
def calculate_expression(expression: str) -> str:
    allowed_chars = set("0123456789+-*/(). %")
    if not set(expression).issubset(allowed_chars):
        return "Error: Expression contains forbidden characters."
    return str(eval(expression, {"__builtins__": None}, {}))

@registry.register_tool(description="Returns user profile details such as name and budget.")
def get_user_profile(user_id: str) -> str:
    profiles = {
        "user_101": json.dumps({"name": "Alice", "portfolio_balance": 50000, "risk_tolerance": "Moderate"}),
        "user_102": json.dumps({"name": "Bob", "portfolio_balance": 15000, "risk_tolerance": "Aggressive"})
    }
    return profiles.get(user_id, f"User '{user_id}' not found.")


# ---------------------------------------------------------
# Test & Demo the Registry
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("TOOL REGISTRY CATALOG PROMPT:")
    print("=" * 70)
    print(registry.get_tool_catalog_prompt())
    print("\n" + "=" * 70)
    print("TESTING TOOL DISPATCH & ERROR HANDLING:")
    print("=" * 70)
    
    # Valid dispatch
    print("\n1. Valid Tool Call:")
    res1 = registry.execute("get_stock_price", ticker="AAPL")
    print(f"Result: {res1}")

    # Valid Calculation
    print("\n2. Calculator Tool Call:")
    res2 = registry.execute("calculate_expression", expression="227.50 * 15")
    print(f"Result: {res2}")

    # Non-existent Tool (Error boundary test)
    print("\n3. Missing Tool Call:")
    res3 = registry.execute("search_web", query="latest news")
    print(f"Result: {res3}")

    # Invalid arguments (Error boundary test)
    print("\n4. Invalid Arguments Call:")
    res4 = registry.execute("get_stock_price", wrong_arg="GOOGL")
    print(f"Result: {res4}")
