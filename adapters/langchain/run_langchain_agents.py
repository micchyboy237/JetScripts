"""
LangChain Agent Example (create_agent + AgentState)
✅ Fixed for LangChain 0.3+ / LangGraph runtime
"""
from typing import TypedDict, List, Dict, Any
from jet.logger import logger
from jet.visualization.langchain.mermaid_graph import render_mermaid_graph
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from jet.file.utils import save_file
import os
import shutil

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = f"{OUTPUT_DIR}/main.log"
logger.basicConfig(filename=log_file)
logger.orange(f"Logs: {log_file}")

class CalcAgentState(TypedDict):
    """Agent state schema containing memory and operation tracking."""
    messages: List[Dict[str, Any]]
    operation_count: int

# Global state for mutation in tools (avoids passing state to tools)
_current_state: CalcAgentState = None

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    if _current_state is not None:
        _current_state["operation_count"] += 1
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    if _current_state is not None:
        _current_state["operation_count"] += 1
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    if _current_state is not None:
        _current_state["operation_count"] += 1
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b (safe divide)."""
    if _current_state is not None:
        _current_state["operation_count"] += 1
    if b == 0:
        return float("inf")
    return a / b

def build_calc_agent():
    """Create a LangChain agent that can perform basic arithmetic."""
    tools = [add, subtract, multiply, divide]
    model = ChatOpenAI(temperature=0, verbosity="high", model="qwen3-instruct-2507:4b", base_url="http://shawn-pc.local:8080/v1")
    
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="You are a precise math assistant. Use tools to compute exactly.",
        debug=True,
    )

    return agent

def main():
    """Demonstrates invoking a LangChain agent with correct state dict."""
    global _current_state
    agent = build_calc_agent()

    # Save agent flowchart
    render_mermaid_graph(agent, output_filename=f"{OUTPUT_DIR}/agent_flow.png")

    _current_state = {
        "messages": [{"role": "user", "content": "Compute 12 / (3 + 1)"}],
        "operation_count": 0,
    }
    result_state = agent.invoke(_current_state)
    print("=== Agent Output ===")
    last_msg = result_state["messages"][-1]
    print(last_msg.content)
    print("\n=== Final State ===")
    print(f"Operation Count: {result_state.get('operation_count', 'N/A')}")

    save_file(result_state["messages"], f"{OUTPUT_DIR}/messages.json")

if __name__ == "__main__":
    main()