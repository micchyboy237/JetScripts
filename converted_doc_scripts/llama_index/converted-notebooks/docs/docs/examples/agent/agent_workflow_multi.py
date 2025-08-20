import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import (
AgentInput,
AgentOutput,
ToolCall,
ToolCallResult,
AgentStream,
)
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
from llama_index.core.settings import Settings
from llama_index.core.workflow import Context
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tavily import AsyncTavilyClient
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Multi-Agent Report Generation with AgentWorkflow

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/agent_workflow_multi.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

In this notebook, we will explore how to use the `AgentWorkflow` class to create multi-agent systems. Specifically, we will create a system that can generate a report on a given topic.

This notebook will assume that you have already either read the [basic agent workflow notebook](https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic) or the [agent workflow documentation](https://docs.llamaindex.ai/en/stable/understanding/agent/).

## Setup

In this example, we will use `MLX` as our LLM. For all LLMs, check out the [examples documentation](https://docs.llamaindex.ai/en/stable/examples/llm/openai/) or [LlamaHub](https://llamahub.ai/?tab=llms) for a list of all supported LLMs and how to install/use them.

If we wanted, each agent could have a different LLM, but for this example, we will use the same LLM for all agents.
"""
logger.info("# Multi-Agent Report Generation with AgentWorkflow")

# %pip install llama-index


llm = MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", api_key="sk-...")

"""
## System Design

Our system will have three agents:

1. A `ResearchAgent` that will search the web for information on the given topic.
2. A `WriteAgent` that will write the report using the information found by the `ResearchAgent`.
3. A `ReviewAgent` that will review the report and provide feedback.

We will use the `AgentWorkflow` class to create a multi-agent system that will execute these agents in order.

While there are many ways to implement this system, in this case, we will use a few tools to help with the research and writing processes.

1. A `web_search` tool to search the web for information on the given topic.
2. A `record_notes` tool to record notes on the given topic.
3. A `write_report` tool to write the report using the information found by the `ResearchAgent`.
4. A `review_report` tool to review the report and provide feedback.

Utilizing the `Context` class, we can pass state between agents, and each agent will have access to the current state of the system.
"""
logger.info("## System Design")

# %pip install tavily-python



async def search_web(query: str) -> str:
    """Useful for using the web to answer questions."""
    client = AsyncTavilyClient(api_key="tvly-...")
    async def run_async_code_2ae1754f():
        return str(await client.search(query))
        return 
     = asyncio.run(run_async_code_2ae1754f())
    logger.success(format_json())


async def record_notes(ctx: Context, notes: str, notes_title: str) -> str:
    """Useful for recording notes on a given topic. Your input should be notes with a title to save the notes under."""
    async def async_func_14():
        async with ctx.store.edit_state() as ctx_state:
            if "research_notes" not in ctx_state["state"]:
                ctx_state["state"]["research_notes"] = {}
            ctx_state["state"]["research_notes"][notes_title] = notes
        return result

    result = asyncio.run(async_func_14())
    logger.success(format_json(result))
    return "Notes recorded."


async def write_report(ctx: Context, report_content: str) -> str:
    """Useful for writing a report on a given topic. Your input should be a markdown formatted report."""
    async def async_func_23():
        async with ctx.store.edit_state() as ctx_state:
            ctx_state["state"]["report_content"] = report_content
        return result

    result = asyncio.run(async_func_23())
    logger.success(format_json(result))
    return "Report written."


async def review_report(ctx: Context, review: str) -> str:
    """Useful for reviewing a report and providing feedback. Your input should be a review of the report."""
    async def async_func_30():
        async with ctx.store.edit_state() as ctx_state:
            ctx_state["state"]["review"] = review
        return result

    result = asyncio.run(async_func_30())
    logger.success(format_json(result))
    return "Report reviewed."

"""
With our tools defined, we can now create our agents.

If the LLM you are using supports tool calling, you can use the `FunctionAgent` class. Otherwise, you can use the `ReActAgent` class.

Here, the name and description of each agent is used so that the system knows what each agent is responsible for and when to hand off control to the next agent.
"""
logger.info("With our tools defined, we can now create our agents.")


research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Useful for searching the web for information on a given topic and recording notes on the topic.",
    system_prompt=(
        "You are the ResearchAgent that can search the web for information on a given topic and record notes on the topic. "
        "Once notes are recorded and you are satisfied, you should hand off control to the WriteAgent to write a report on the topic. "
        "You should have at least some notes on a topic before handing off control to the WriteAgent."
    ),
    llm=llm,
    tools=[search_web, record_notes],
    can_handoff_to=["WriteAgent"],
)

write_agent = FunctionAgent(
    name="WriteAgent",
    description="Useful for writing a report on a given topic.",
    system_prompt=(
        "You are the WriteAgent that can write a report on a given topic. "
        "Your report should be in a markdown format. The content should be grounded in the research notes. "
        "Once the report is written, you should get feedback at least once from the ReviewAgent."
    ),
    llm=llm,
    tools=[write_report],
    can_handoff_to=["ReviewAgent", "ResearchAgent"],
)

review_agent = FunctionAgent(
    name="ReviewAgent",
    description="Useful for reviewing a report and providing feedback.",
    system_prompt=(
        "You are the ReviewAgent that can review the write report and provide feedback. "
        "Your review should either approve the current report or request changes for the WriteAgent to implement. "
        "If you have feedback that requires changes, you should hand off control to the WriteAgent to implement the changes after submitting the review."
    ),
    llm=llm,
    tools=[review_report],
    can_handoff_to=["WriteAgent"],
)

"""
## Running the Workflow

With our agents defined, we can create our `AgentWorkflow` and run it.
"""
logger.info("## Running the Workflow")


agent_workflow = AgentWorkflow(
    agents=[research_agent, write_agent, review_agent],
    root_agent=research_agent.name,
    initial_state={
        "research_notes": {},
        "report_content": "Not written yet.",
        "review": "Review required.",
    },
)

"""
As the workflow is running, we will stream the events to get an idea of what is happening under the hood.
"""
logger.info("As the workflow is running, we will stream the events to get an idea of what is happening under the hood.")


handler = agent_workflow.run(
    user_msg=(
        "Write me a report on the history of the internet. "
        "Briefly describe the history of the internet, including the development of the internet, the development of the web, "
        "and the development of the internet in the 21st century."
    )
)

current_agent = None
current_tool_calls = ""
async for event in handler.stream_events():
    if (
        hasattr(event, "current_agent_name")
        and event.current_agent_name != current_agent
    ):
        current_agent = event.current_agent_name
        logger.debug(f"\n{'='*50}")
        logger.debug(f"🤖 Agent: {current_agent}")
        logger.debug(f"{'='*50}\n")

    elif isinstance(event, AgentOutput):
        if event.response.content:
            logger.debug("📤 Output:", event.response.content)
        if event.tool_calls:
            logger.debug(
                "🛠️  Planning to use tools:",
                [call.tool_name for call in event.tool_calls],
            )
    elif isinstance(event, ToolCallResult):
        logger.debug(f"🔧 Tool Result ({event.tool_name}):")
        logger.debug(f"  Arguments: {event.tool_kwargs}")
        logger.debug(f"  Output: {event.tool_output}")
    elif isinstance(event, ToolCall):
        logger.debug(f"🔨 Calling Tool: {event.tool_name}")
        logger.debug(f"  With arguments: {event.tool_kwargs}")

"""
Now, we can retrieve the final report in the system for ourselves.
"""
logger.info("Now, we can retrieve the final report in the system for ourselves.")

async def run_async_code_fadf6923():
    async def run_async_code_6b347448():
        state = await handler.ctx.store.get("state")
        return state
    state = asyncio.run(run_async_code_6b347448())
    logger.success(format_json(state))
    return state
state = asyncio.run(run_async_code_fadf6923())
logger.success(format_json(state))
logger.debug(state["report_content"])

logger.info("\n\n[DONE]", bright=True)