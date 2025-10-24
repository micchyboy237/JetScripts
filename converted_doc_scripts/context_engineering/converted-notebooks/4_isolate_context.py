from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_sandbox import PyodideSandboxTool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from utils import format_messages
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

"""
# Isolating Context

*Isolating context involves splitting it up to help an agent perform a task.*

![Screenshot 2025-07-09 at 2.28.19 PM.png](attachment:96e3f693-02e4-47c2-9f03-3a1c3146c84a.png)

## Multi-Agent

One of the most popular and intuitive ways to isolate context is to split it across sub-agents. A motivation for the Ollama [Swarm](https://github.com/ollama/swarm) library was “[separation of concerns](https://ollama.github.io/ollama-agents-python/ref/agent/)”, where a team of agents can handle sub-tasks. Each agent has a specific set of tools, instructions, and its own context window.

Ollama’s [multi-agent researcher](https://www.anthropic.com/engineering/built-multi-agent-research-system) makes a clear case for the benefit of this: many agents with isolated contexts outperformed single-agent by 90.2%, largely because each subagent context window can be allocated to a more narrow sub-task. As the blog said:

> [Subagents operate] in parallel with their own context windows, exploring different aspects of the question simultaneously. 

![image (3).webp](attachment:21c82d0f-1baa-48c1-a13a-628b2782d836.webp)

Of course, the challenge with multi-agent include token use (e.g., [15× more tokens](https://www.anthropic.com/engineering/built-multi-agent-research-system) than chat), the need for careful [prompt engineering](https://www.anthropic.com/engineering/built-multi-agent-research-system) to plan sub-agent work, and coordination of sub-agents.

### Multi-Agent in LangGraph

LangGraph supports multi-agent systems. A popular and intuitive way to implement this is the [supervisor](https://github.com/langchain-ai/langgraph-supervisor-py) architecture, which is what is used in Ollama's [multi-agent researcher](https://www.anthropic.com/engineering/built-multi-agent-research-system). This allows the supervisor to delegate tasks to sub-agents, each with their own context window.

"""
logger.info("# Isolating Context")


llm = init_chat_model("ollama:claude-sonnet-4-20250514", temperature=0)


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


math_agent = create_react_agent(
    model=llm,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time."
)

research_agent = create_react_agent(
    model=llm,
    tools=[web_search],
    name="research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math."
)

workflow = create_supervisor(
    [research_agent, math_agent],
    model=llm,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    )
)

app = workflow.compile()

result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "what's the combined headcount of the FAANG companies in 2024?"
        }
    ]
})


format_messages(result['messages'])

"""
### Learn more

* **LangGraph Swarm** - https://github.com/langchain-ai/langgraph-swarm-py

LangGraph Swarm is a Python library for creating multi-agent AI systems with dynamic collaboration capabilities. Key features include agents that can dynamically hand off control based on specialization while maintaining conversation context between transitions. The library supports customizable handoff tools between agents, streaming, short-term and long-term memory, and human-in-the-loop interactions. Built on the LangGraph framework, it enables creating flexible, context-aware multi-agent systems where different AI agents can collaborate and seamlessly transfer conversation control based on their unique capabilities. Installation is simple with `pip install langgraph-swarm`.

* [See](https://www.youtube.com/watch?v=4nZl32FwU-o) [these](https://www.youtube.com/watch?v=JeyDrn1dSUQ) [videos](https://www.youtube.com/watch?v=B_0TNuYi56w) for more detail on on multi-agent systems.


## Sandboxed Environment

HuggingFace’s [deep researcher](https://huggingface.co/blog/open-deep-research#:~:text=From%20building%20,it%20can%20still%20use%20it) shows another interesting example of context isolation. Most agents use [tool calling APIs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview), which return JSON objects (tool arguments) that can be passed to tools (e.g., a search API) to get tool feedback (e.g., search results). HuggingFace uses a [CodeAgent](https://huggingface.co/papers/2402.01030), which outputs code to invoke tools. The code then runs in a [sandbox](https://e2b.dev/). Selected context (e.g., return values) from code execution is then passed back to the LLM.

This allows context to be isolated in the environment, outside of the LLM context window. Hugging Face noted that this is a great way to isolate token-heavy objects from the LLM:

> [Code Agents allow for] a better handling of state … Need to store this image / audio / other for later use? No problem, just assign it as a variable in your state and you [use it later].

### Sandboxed Environment in LangGraph

It's pretty easy to use Sandboxes with LangGraph agents. [LangChain Sandbox](https://github.com/langchain-ai/langchain-sandbox) provides a secure environment for executing untrusted Python code. It leverages Pyodide (Python compiled to WebAssembly) to run Python code in a sandboxed environment. This can simply be used as a tool in a LangGraph agent.

> NOTE: Install Deno (required): https://docs.deno.com/runtime/getting_started/installation/

"""
logger.info("### Learn more")

tool = PyodideSandboxTool()
result = await tool.ainvoke("logger.debug('Hello, world!')")
logger.success(format_json(result))



tool = PyodideSandboxTool(
    allow_net=True
)

agent = create_react_agent(
    "ollama:claude-3-7-sonnet-latest",
    tools=[tool],
)

result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's 5 + 7?"}]},
    )
logger.success(format_json(result))

format_messages(result['messages'])

"""
### State 

An agent’s runtime state object can also be a great way to isolate context. This can serve the same purpose as sandboxing. A state object can be designed with a schema (e.g., a Pydantic model) that has various fields that context can be written to. One field of the schema (e.g., messages) can be exposed to the LLM at each turn of the agent, but the schema can isolate information in other fields for more selective use. 

### State Isolation in LangGraph

LangGraph is designed around a [state](https://langchain-ai.github.io/langgraph/concepts/low_level/#state) object, allowing you to design a state schema and access different fields of that schema across trajectory of your agent. For example, you can easily store context from tool calls in certain fields of your state object, isolating from the LLM until that context is required. In these notebooks, you've seen numerous example of this.

"""
logger.info("### State")

logger.info("\n\n[DONE]", bright=True)