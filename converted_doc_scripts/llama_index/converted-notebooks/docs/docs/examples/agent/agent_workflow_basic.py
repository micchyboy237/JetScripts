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
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.core.workflow import (
Context,
InputRequiredEvent,
HumanResponseEvent,
)
from llama_index.core.workflow import Context
from llama_index.core.workflow import JsonPickleSerializer, JsonSerializer
from llama_index.core.workflow import JsonSerializer
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
# FunctionAgent / AgentWorkflow Basic Introduction

The `AgentWorkflow` is an orchestrator for running a system of one or more agents. In this example, we'll create a simple workflow with a single `FunctionAgent`, and use that to cover the basic functionality.
"""
logger.info("# FunctionAgent / AgentWorkflow Basic Introduction")

# %pip install llama-index

"""
## Setup

In this example, we will use `MLX` as our LLM. For all LLMs, check out the [examples documentation](https://docs.llamaindex.ai/en/stable/examples/llm/openai/) or [LlamaHub](https://llamahub.ai/?tab=llms) for a list of all supported LLMs and how to install/use them.
"""
logger.info("## Setup")


llm = MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit-mini", api_key="sk-...")

"""
To make our agent more useful, we can give it tools/actions to use. In this case, we'll use Tavily to implement a tool that can search the web for information. You can get a free API key from [Tavily](https://tavily.com/).
"""
logger.info("To make our agent more useful, we can give it tools/actions to use. In this case, we'll use Tavily to implement a tool that can search the web for information. You can get a free API key from [Tavily](https://tavily.com/).")

# %pip install tavily-python

"""
When creating a tool, its very important to:
- give the tool a proper name and docstring/description. The LLM uses this to understand what the tool does.
- annotate the types. This helps the LLM understand the expected input and output types.
- use async when possible, since this will make the workflow more efficient.
"""
logger.info("When creating a tool, its very important to:")



async def search_web(query: str) -> str:
    """Useful for using the web to answer questions."""
    client = AsyncTavilyClient(api_key="tvly-...")
    async def run_async_code_2ae1754f():
        return str(await client.search(query))
        return 
     = asyncio.run(run_async_code_2ae1754f())
    logger.success(format_json())

"""
With the tool and and LLM defined, we can create an `AgentWorkflow` that uses the tool.
"""
logger.info("With the tool and and LLM defined, we can create an `AgentWorkflow` that uses the tool.")


agent = FunctionAgent(
    tools=[search_web],
    llm=llm,
    system_prompt="You are a helpful assistant that can search the web for information.",
)

"""
## Running the Agent

Now that our agent is created, we can run it!
"""
logger.info("## Running the Agent")

async def run_async_code_70207cd7():
    async def run_async_code_1cec56a8():
        response = await agent.run(user_msg="What is the weather in San Francisco?")
        return response
    response = asyncio.run(run_async_code_1cec56a8())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_70207cd7())
logger.success(format_json(response))
logger.debug(str(response))

"""
The above is the equivalent of the following of using `AgentWorkflow` with a single `FunctionAgent`:
"""
logger.info("The above is the equivalent of the following of using `AgentWorkflow` with a single `FunctionAgent`:")


workflow = AgentWorkflow(agents=[agent])

async def run_async_code_c35c9f3a():
    async def run_async_code_6aae748b():
        response = await workflow.run(user_msg="What is the weather in San Francisco?")
        return response
    response = asyncio.run(run_async_code_6aae748b())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_c35c9f3a())
logger.success(format_json(response))

"""
If you were creating a workflow with multiple agents, you can pass in a list of agents to the `AgentWorkflow` constructor. Learn more in our [multi-agent workflow example](https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agent/).

## Maintaining State

By default, the `FunctionAgent` will maintain stateless between runs. This means that the agent will not have any memory of previous runs.

To maintain state, we need to keep track of the previous state. Since the `FunctionAgent` is running in a  `Workflow`, the state is stored in the `Context`. This can be passed between runs to maintain state and history.
"""
logger.info("## Maintaining State")


ctx = Context(agent)

async def async_func_4():
    response = await agent.run(
        user_msg="My name is Logan, nice to meet you!", ctx=ctx
    )
    return response
response = asyncio.run(async_func_4())
logger.success(format_json(response))
logger.debug(str(response))

async def run_async_code_443384f5():
    async def run_async_code_7e525c02():
        response = await agent.run(user_msg="What is my name?", ctx=ctx)
        return response
    response = asyncio.run(run_async_code_7e525c02())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_443384f5())
logger.success(format_json(response))
logger.debug(str(response))

"""
The context is serializable, so it can be saved to a database, file, etc. and loaded back in later. 

The `JsonSerializer` is a simple serializer that uses `json.dumps` and `json.loads` to serialize and deserialize the context.

The `JsonPickleSerializer` is a serializer that uses `pickle` to serialize and deserialize the context. If you have objects in your context that are not serializable, you can use this serializer.
"""
logger.info("The context is serializable, so it can be saved to a database, file, etc. and loaded back in later.")


ctx_dict = ctx.to_dict(serializer=JsonSerializer())

restored_ctx = Context.from_dict(agent, ctx_dict, serializer=JsonSerializer())

async def async_func_6():
    response = await agent.run(
        user_msg="Do you still remember my name?", ctx=restored_ctx
    )
    return response
response = asyncio.run(async_func_6())
logger.success(format_json(response))
logger.debug(str(response))

"""
## Streaming

The `AgentWorkflow`/`FunctionAgent` also supports streaming. Since the `AgentWorkflow` is a `Workflow`, it can be streamed like any other `Workflow`. This works by using the handler that is returned from the workflow. There are a few key events that are streamed, feel free to explore below.

If you only want to stream the LLM output, you can use the `AgentStream` events.
"""
logger.info("## Streaming")


handler = agent.run(user_msg="What is the weather in Saskatoon?")

async for event in handler.stream_events():
    if isinstance(event, AgentStream):
        logger.debug(event.delta, end="", flush=True)

"""
## Tools and State

Tools can also be defined that have access to the workflow context. This means you can set and retrieve variables from the context and use them in the tool or between tools.

**Note:** The `Context` parameter should be the first parameter of the tool.
"""
logger.info("## Tools and State")



async def set_name(ctx: Context, name: str) -> str:
    async def async_func_4():
        async with ctx.store.edit_state() as ctx_state:
            ctx_state["state"]["name"] = name
        return result

    result = asyncio.run(async_func_4())
    logger.success(format_json(result))
    return f"Name set to {name}"


agent = FunctionAgent(
    tools=[set_name],
    llm=llm,
    system_prompt="You are a helpful assistant that can set a name.",
    initial_state={"name": "unset"},
)

ctx = Context(agent)

async def run_async_code_78e2318c():
    async def run_async_code_73a34dda():
        response = await agent.run(user_msg="My name is Logan", ctx=ctx)
        return response
    response = asyncio.run(run_async_code_73a34dda())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_78e2318c())
logger.success(format_json(response))
logger.debug(str(response))

async def run_async_code_d2a788b4():
    async def run_async_code_6363ca33():
        state = await ctx.store.get("state")
        return state
    state = asyncio.run(run_async_code_6363ca33())
    logger.success(format_json(state))
    return state
state = asyncio.run(run_async_code_d2a788b4())
logger.success(format_json(state))
logger.debug(state["name"])

"""
## Human in the Loop

Tools can also be defined that involve a human in the loop. This is useful for tasks that require human input, such as confirming a tool call or providing feedback.

Using workflow events, we can emit events that require a response from the user. Here, we use the built-in `InputRequiredEvent` and `HumanResponseEvent` to handle the human in the loop, but you can also define your own events.

`wait_for_event` will emit the `waiter_event` and wait until it sees the `HumanResponseEvent` with the specified `requirements`. The `waiter_id` is used to ensure that we only send one `waiter_event` for each `waiter_id`.
"""
logger.info("## Human in the Loop")



async def dangerous_task(ctx: Context) -> str:
    """A dangerous task that requires human confirmation."""

    question = "Are you sure you want to proceed?"
    async def async_func_11():
        response = await ctx.wait_for_event(
            HumanResponseEvent,
            waiter_id=question,
            waiter_event=InputRequiredEvent(
                prefix=question,
                user_name="Logan",
            ),
            requirements={"user_name": "Logan"},
        )
        return response
    response = asyncio.run(async_func_11())
    logger.success(format_json(response))
    if response.response == "yes":
        return "Dangerous task completed successfully."
    else:
        return "Dangerous task aborted."


agent = FunctionAgent(
    tools=[dangerous_task],
    llm=llm,
    system_prompt="You are a helpful assistant that can perform dangerous tasks.",
)

handler = agent.run(user_msg="I want to proceed with the dangerous task.")

async for event in handler.stream_events():
    if isinstance(event, InputRequiredEvent):
        response = input(event.prefix).strip().lower()
        handler.ctx.send_event(
            HumanResponseEvent(
                response=response,
                user_name=event.user_name,
            )
        )

async def run_async_code_78ff5c2b():
    async def run_async_code_2cbcd794():
        response = await handler
        return response
    response = asyncio.run(run_async_code_2cbcd794())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_78ff5c2b())
logger.success(format_json(response))
logger.debug(str(response))

"""
In production scenarios, you might handle human-in-the-loop over a websocket or multiple API requests.

As mentioned before, the `Context` object is serializable, and this means we can also save the workflow mid-run and restore it later. 

**NOTE:** Any functions/steps that were in-progress will start from the beginning when the workflow is restored.
"""
logger.info("In production scenarios, you might handle human-in-the-loop over a websocket or multiple API requests.")


handler = agent.run(user_msg="I want to proceed with the dangerous task.")

input_ev = None
async for event in handler.stream_events():
    if isinstance(event, InputRequiredEvent):
        input_ev = event
        break

ctx_dict = handler.ctx.to_dict(serializer=JsonSerializer())

response_str = input(input_ev.prefix).strip().lower()

restored_ctx = Context.from_dict(agent, ctx_dict, serializer=JsonSerializer())

handler = agent.run(ctx=restored_ctx)
handler.ctx.send_event(
    HumanResponseEvent(
        response=response_str,
        user_name=input_ev.user_name,
    )
)
async def run_async_code_78ff5c2b():
    async def run_async_code_2cbcd794():
        response = await handler
        return response
    response = asyncio.run(run_async_code_2cbcd794())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_78ff5c2b())
logger.success(format_json(response))
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)