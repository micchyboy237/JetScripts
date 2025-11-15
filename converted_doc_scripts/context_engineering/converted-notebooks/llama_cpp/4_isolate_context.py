import json
from pathlib import Path
import shutil
from jet.visualization.langchain.mermaid_graph import render_mermaid_graph
from rich.console import Console
from rich.pretty import pprint
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from jet.visualization.terminal import display_iterm2_image
from jet.adapters.langchain.chat_agent_utils import build_agent
from jet.adapters.llama_cpp.tokens import count_tokens

import os
from jet.logger import logger

BASE_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(BASE_OUTPUT_DIR, ignore_errors=True)

console = Console()

def add(a: float, b: float) -> float:
    """Add two numbers.
    Args:
        a: First number
        b: Second number
    Returns:
        Sum of a and b
    """
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers.
    Args:
        a: First number
        b: Second number
    Returns:
        Product of a and b
    """
    return a * b

def web_search(query: str) -> str:
    """Mock web search function that returns FAANG company headcounts.
    Args:
        query: Search query (unused in this mock)
    Returns:
        Static information about FAANG company headcounts
    """
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )

llm_math = ChatOpenAI(
    model="qwen3-instruct-2507:4b",
    temperature=0.0,
    base_url="http://shawn-pc.local:8080/v1",
    verbosity="high",
)

llm_research = ChatOpenAI(
    model="qwen3-instruct-2507:4b",
    temperature=0.0,
    base_url="http://shawn-pc.local:8080/v1",
    verbosity="high",
)

llm_supervisor = ChatOpenAI(
    model="qwen3-instruct-2507:4b",
    temperature=0.0,
    base_url="http://shawn-pc.local:8080/v1",
    verbosity="high",
)

# ------------------------------------------------------------------
# `create_supervisor` requires each sub-agent to have a `.name`.
# `build_agent` does not expose `name`, so we:
#   1. create the raw agent with `create_react_agent`,
#   2. set `.name`,
#   3. re-wrap with `build_agent` (adds logging/middleware),
#   4. copy the name back.
# ------------------------------------------------------------------

# ---- Math agent ----------------------------------------------------
math_prompt = "You are a math expert. Always use one tool at a time."
math_agent = build_agent(
    tools=[add, multiply],
    model=llm_math,
    system_prompt=math_prompt,
    name="math_expert",
)

# ---- Research agent ------------------------------------------------
research_prompt = "You are a world class researcher with access to web search. Do not do any math."
research_agent = build_agent(
    tools=[web_search],
    model=llm_research,
    system_prompt=research_prompt,
    name="research_expert",
)

# Supervisor with context isolation via token counting
workflow = create_supervisor(
    [research_agent, math_agent],
    model=llm_supervisor,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. For math problems, use math_agent. "
        "Estimate context size before routing."
    ),
    supervisor_name="supervisor_jet"
)

from langchain_sandbox import PyodideSandboxTool
from langgraph.prebuilt import create_react_agent as create_react_agent_raw

llm_sandbox = ChatOpenAI(
    model="qwen3-instruct-2507:4b",
    temperature=0.0,
    base_url="http://shawn-pc.local:8080/v1",
    verbosity="high",
)

tool = PyodideSandboxTool(allow_net=True)
sandbox_agent = create_react_agent_raw(
    model=llm_sandbox,
    tools=[tool],
)

def example_1_supervisor_routing():
    """Example 1: Supervisor routes between research and math agents."""
    example_dir = Path(BASE_OUTPUT_DIR) / "example_1_supervisor_routing"
    example_dir.mkdir(parents=True, exist_ok=True)
    log_file = example_dir / "main.log"
    logger.basicConfig(filename=log_file, level=logger.INFO, force=True)
    logger.orange(f"Example 1 logs: {log_file}")

    app = workflow.compile()

    png_path = example_dir / "supervisor_graph.png"
    png_data = render_mermaid_graph(app, output_filename=str(png_path))
    display_iterm2_image(png_data)

    query = "what's the combined headcount of the FAANG companies in 2024?"
    query_tokens = count_tokens([query], model="qwen3-instruct-2507:4b")
    logger.log(f"User query tokens: {query_tokens}", colors=["INFO", "YELLOW"])

    result = app.invoke({"messages": [{"role": "user", "content": query}]})
    (example_dir / "result.json").write_text(json.dumps(result, indent=2))

    console.print("\n[bold blue]Example 1 - Multi-Agent Workflow State:[/bold blue]")
    pprint(result)

def example_2_sandbox_execution():
    """Example 2: React agent with Pyodide sandbox for safe code execution."""
    example_dir = Path(BASE_OUTPUT_DIR) / "example_2_sandbox_execution"
    example_dir.mkdir(parents=True, exist_ok=True)
    log_file = example_dir / "main.log"
    logger.basicConfig(filename=log_file, level=logger.INFO, force=True)
    logger.orange(f"Example 2 logs: {log_file}")

    code_query = "what's 5 + 7?"
    code_tokens = count_tokens([code_query], model="qwen3-instruct-2507:4b")
    logger.log(f"Sandbox query tokens: {code_tokens}", colors=["INFO", "YELLOW"])

    result = sandbox_agent.invoke({"messages": [{"role": "user", "content": code_query}]})
    (example_dir / "code_result.json").write_text(json.dumps(result, indent=2))

    png_path = example_dir / "sandbox_agent_graph.png"
    final_png = render_mermaid_graph(sandbox_agent, output_filename=str(png_path))
    display_iterm2_image(final_png)

    console.print("\n[bold blue]Example 2 - React agent with sandbox result:[/bold blue]")
    pprint(result)

if __name__ == "__main__":
    console.print("[bold magenta]Running 4_isolate_context.py examples...[/bold magenta]")
    example_1_supervisor_routing()
    example_2_sandbox_execution()
    console.print("[bold green]All examples completed.[/bold green]")