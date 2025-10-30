"""
LangChain Agent Example (create_agent + AgentState)
✅ Fixed for LangChain 0.3+ / LangGraph runtime
"""
from typing import TypedDict, List, Dict, Any
from typing import Callable, Awaitable
from jet.transformers.formatters import format_json
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langgraph.prebuilt.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from jet.logger import logger, DEFAULT_LOGGER, CustomLogger
from jet.visualization.langchain.mermaid_graph import render_mermaid_graph
from jet.file.utils import save_file
import logging
import os
import shutil

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = f"{OUTPUT_DIR}/main.log"
logger.basicConfig(DEFAULT_LOGGER, filename=log_file, level=logging.DEBUG)
logger.orange(f"Main logs: {log_file}")

agent_log_file = f"{OUTPUT_DIR}/agent.log"
agent_logger = CustomLogger("agent", filename=agent_log_file)
logger.orange(f"Agent logs: {agent_log_file}")

model_log_file = f"{OUTPUT_DIR}/model.log"
model_logger = CustomLogger("model", filename=model_log_file)
logger.orange(f"Model logs: {model_log_file}")

tool_log_file = f"{OUTPUT_DIR}/tool.log"
tool_logger = CustomLogger("tool", filename=tool_log_file)
logger.orange(f"Tool logs: {tool_log_file}")

# tool_log_file = f"{OUTPUT_DIR}/tools.log"
# tool_logger = CustomLogger("tools", filename=tool_log_file, level=logging.DEBUG)
# logger.orange(f"Tool logs: {tool_log_file}")

# ─────────────────────────────────────────────────────────────────────────────
#  TOOL-CALL LOGGING MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────
class ToolCallLoggingMiddleware(AgentMiddleware):
    """Logs start/end of every tool call (inputs → result)."""

    def __init__(self) -> None:
        super().__init__()

    def before_agent(self, state, runtime):
        agent_logger.info(
            "[BEFORE AGENT] (State=%s)", format_json(state)
        )

    def after_agent(self, state, runtime):
        agent_logger.teal(
            "[AFTER AGENT] (State=%s)", format_json(state)
        )

    # ── SYNC ─────────────────────────────────────────────────────────────────
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_name = request.tool_call.get("name", "unknown")
        tool_id = request.tool_call.get("id", "unknown")
        args = request.tool_call.get("args", {})

        tool_logger.info(
            "[TOOL START] %s\nid: %s\nargs: %s", tool_name, tool_id, format_json(args)
        )
        result = handler(request)
        tool_logger.teal(
            "[TOOL END] %s\nid: %s\nresult: %s", tool_name, tool_id, format_json(result)
        )
        return result

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ToolMessage | Command:
        model_settings = request.model_settings
        system_prompt = request.system_prompt
        messages = request.messages
        tool_choice = request.tool_choice
        tools = request.tools
        agent_state = request.state

        model_logger.info(
            "[MODEL START]\n%s", format_json({
                "model_settings": model_settings,
                "system_prompt": system_prompt,
                "messages": messages,
                "tool_choice": tool_choice,
                "tools": tools,
                "agent_state": agent_state,
            })
        )
        result = handler(request)
        model_logger.teal(
            "[MODEL END] result=%s", result
        )
        return result

    # ── ASYNC ───────────────────────────────────────────────────────────────
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        tool_name = request.tool_call.get("name", "unknown")
        tool_id = request.tool_call.get("id", "unknown")
        args = request.tool_call.get("args", {})

        tool_logger.info(
            "[TOOL START] %s (id=%s) args=%s", tool_name, tool_id, args
        )
        result = await handler(request)
        tool_logger.teal(
            "[TOOL END] %s (id=%s) result=%s", tool_name, tool_id, result
        )
        return result

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ToolMessage | Command:
        model_settings = request.model_settings
        system_prompt = request.system_prompt
        messages = request.messages
        tool_choice = request.tool_choice
        tools = request.tools
        agent_state = request.state

        model_logger.info(
            "[MODEL START]\n%s", format_json({
                "model_settings": model_settings,
                "system_prompt": system_prompt,
                "messages": messages,
                "tool_choice": tool_choice,
                "tools": tools,
                "agent_state": agent_state,
            })
        )
        result = await handler(request)
        model_logger.teal(
            "[MODEL END] result=%s", result
        )
        return result

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
        middleware=[ToolCallLoggingMiddleware()],
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