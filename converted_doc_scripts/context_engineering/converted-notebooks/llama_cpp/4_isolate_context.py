from rich.console import Console
from rich.pretty import pprint
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from jet.visualization.terminal import display_iterm2_image
from jet.adapters.langchain.chat_agent_utils import build_agent
from jet.adapters.llama_cpp.tokens import count_tokens

import os
import shutil
from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = f"{OUTPUT_DIR}/main.log"
logger.basicConfig(filename=log_file)
logger.orange(f"Main logs: {log_file}")

console = Console()

llm = ChatOpenAI(
    model="qwen3-instruct-2507:4b",
    temperature=0.0,
    base_url="http://shawn-pc.local:8080/v1",
    verbosity="high",
)

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

# ------------------------------------------------------------------
# `create_supervisor` requires each sub-agent to have a `.name`.
# `build_agent` does not expose `name`, so we:
#   1. create the raw agent with `create_react_agent`,
#   2. set `.name`,
#   3. re-wrap with `build_agent` (adds logging/middleware),
#   4. copy the name back.
# ------------------------------------------------------------------
from langgraph.prebuilt import create_react_agent as _create_react_agent

# ---- Math agent ----------------------------------------------------
math_raw = _create_react_agent(
    model=llm,
    tools=[add, multiply],
    prompt="You are a math expert. Always use one tool at a time.",
)
math_raw.name = "math_expert"

math_prompt = "You are a math expert. Always use one tool at a time."
math_agent = build_agent(
    tools=[add, multiply],
    model=llm,
    system_prompt=math_prompt,
)
math_agent.name = math_raw.name

# ---- Research agent ------------------------------------------------
research_raw = _create_react_agent(
    model=llm,
    tools=[web_search],
    prompt="You are a world class researcher with access to web search. Do not do any math.",
)
research_raw.name = "research_expert"

research_prompt = "You are a world class researcher with access to web search. Do not do any math."
research_agent = build_agent(
    tools=[web_search],
    model=llm,
    system_prompt=research_prompt,
)
research_agent.name = research_raw.name

# Supervisor with context isolation via token counting
workflow = create_supervisor(
    [research_agent, math_agent],
    model=llm,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. For math problems, use math_agent. "
        "Estimate context size before routing."
    )
)

app = workflow.compile()
png_data = app.get_graph().draw_mermaid_png()
display_iterm2_image(png_data)

# Context-aware invocation with token logging
query = "what's the combined headcount of the FAANG companies in 2024?"
query_tokens = count_tokens([query], model="qwen3-instruct-2507:4b")
logger.log(f"User query tokens: {query_tokens}", colors=["INFO", "YELLOW"])

result = app.invoke({
    "messages": [
        {"role": "user", "content": query}
    ]
})

console.print("\n[bold blue]Multi-Agent Workflow State:[/bold blue]")
pprint(result)

# Sandbox tool with async isolation
from langchain_sandbox import PyodideSandboxTool
from langgraph.prebuilt import create_react_agent as create_react_agent_raw

tool = PyodideSandboxTool(allow_net=True)
sandbox_agent = create_react_agent_raw(
    model=llm,
    tools=[tool],
)

code_query = "what's 5 + 7?"
code_tokens = count_tokens([code_query], model="qwen3-instruct-2507:4b")
logger.log(f"Sandbox query tokens: {code_tokens}", colors=["INFO", "YELLOW"])

result = sandbox_agent.invoke(
    {"messages": [{"role": "user", "content": code_query}]},
)

console.print("\n[bold blue]React agent with sandbox tool result:[/bold blue]")
pprint(result)

# Final graph visualization
final_png = sandbox_agent.get_graph().draw_mermaid_png()
display_iterm2_image(final_png)