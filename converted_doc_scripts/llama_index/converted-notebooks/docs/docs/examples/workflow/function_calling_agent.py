import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.settings import Settings
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.tools.types import BaseTool
from llama_index.core.workflow import (
Context,
Workflow,
StartEvent,
StopEvent,
step,
)
from llama_index.core.workflow import Context
from llama_index.core.workflow import Event
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Any, List
import asyncio
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
# Workflow for a Function Calling Agent

This notebook walks through setting up a `Workflow` to construct a function calling agent from scratch.

Function calling agents work by using an LLM that supports tools/functions in its API (MLX, Ollama, Anthropic, etc.) to call functions an use tools.

Our workflow will be stateful with memory, and will be able to call the LLM to select tools and process incoming user messages.
"""
logger.info("# Workflow for a Function Calling Agent")

# !pip install -U llama-index


# os.environ["OPENAI_API_KEY"] = "sk-proj-..."

"""
### [Optional] Set up observability with Llamatrace

Set up tracing to visualize each step in the workflow.

Since workflows are async first, this all runs fine in a notebook. If you were running in your own code, you would want to use `asyncio.run()` to start an async event loop if one isn't already running.

```python
async def main():
    <async code>

if __name__ == "__main__":
    asyncio.run(main())
```

## Designing the Workflow

An agent consists of several steps
1. Handling the latest incoming user message, including adding to memory and getting the latest chat history
2. Calling the LLM with tools + chat history
3. Parsing out tool calls (if any)
4. If there are tool calls, call them, and loop until there are none
5. When there is no tool calls, return the LLM response

### The Workflow Events

To handle these steps, we need to define a few events:
1. An event to handle new messages and prepare the chat history
2. An event to handle streaming responses
3. An event to trigger tool calls
4. An event to handle the results of tool calls

The other steps will use the built-in `StartEvent` and `StopEvent` events.
"""
logger.info("### [Optional] Set up observability with Llamatrace")



class InputEvent(Event):
    input: list[ChatMessage]


class StreamEvent(Event):
    delta: str


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class FunctionOutputEvent(Event):
    output: ToolOutput

"""
### The Workflow Itself

With our events defined, we can construct our workflow and steps. 

Note that the workflow automatically validates itself using type annotations, so the type annotations on our steps are very helpful!
"""
logger.info("### The Workflow Itself")




class FuncationCallingAgent(Workflow):
    def __init__(
        self,
        *args: Any,
        llm: FunctionCallingLLM | None = None,
        tools: List[BaseTool] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []

        self.llm = llm or MLXLlamaIndexLLMAdapter()
        assert self.llm.metadata.is_function_calling_model

    @step
    async def prepare_chat_history(
        self, ctx: Context, ev: StartEvent
    ) -> InputEvent:
        async def run_async_code_74de6a9e():
            await ctx.store.set("sources", [])
            return 
         = asyncio.run(run_async_code_74de6a9e())
        logger.success(format_json())

        async def run_async_code_4f414010():
            async def run_async_code_7b91d640():
                memory = await ctx.store.get("memory", default=None)
                return memory
            memory = asyncio.run(run_async_code_7b91d640())
            logger.success(format_json(memory))
            return memory
        memory = asyncio.run(run_async_code_4f414010())
        logger.success(format_json(memory))
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)

        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        memory.put(user_msg)

        chat_history = memory.get()

        async def run_async_code_bbf63658():
            await ctx.store.set("memory", memory)
            return 
         = asyncio.run(run_async_code_bbf63658())
        logger.success(format_json())

        return InputEvent(input=chat_history)

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        chat_history = ev.input

        async def async_func_55():
            response_stream = self.llm.stream_chat_with_tools(
                self.tools, chat_history=chat_history
            )
            return response_stream
        response_stream = asyncio.run(async_func_55())
        logger.success(format_json(response_stream))
        async for response in response_stream:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))

        async def run_async_code_7cb3b9e4():
            async def run_async_code_94b374ec():
                memory = await ctx.store.get("memory")
                return memory
            memory = asyncio.run(run_async_code_94b374ec())
            logger.success(format_json(memory))
            return memory
        memory = asyncio.run(run_async_code_7cb3b9e4())
        logger.success(format_json(memory))
        memory.put(response.message)
        async def run_async_code_bbf63658():
            await ctx.store.set("memory", memory)
            return 
         = asyncio.run(run_async_code_bbf63658())
        logger.success(format_json())

        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )

        if not tool_calls:
            async def run_async_code_b3447605():
                async def run_async_code_54ba06e9():
                    sources = await ctx.store.get("sources", default=[])
                    return sources
                sources = asyncio.run(run_async_code_54ba06e9())
                logger.success(format_json(sources))
                return sources
            sources = asyncio.run(run_async_code_b3447605())
            logger.success(format_json(sources))
            return StopEvent(
                result={"response": response, "sources": [*sources]}
            )
        else:
            return ToolCallEvent(tool_calls=tool_calls)

    @step
    async def handle_tool_calls(
        self, ctx: Context, ev: ToolCallEvent
    ) -> InputEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        tool_msgs = []
        async def run_async_code_dc23cd7a():
            async def run_async_code_b3447605():
                sources = await ctx.store.get("sources", default=[])
                return sources
            sources = asyncio.run(run_async_code_b3447605())
            logger.success(format_json(sources))
            return sources
        sources = asyncio.run(run_async_code_dc23cd7a())
        logger.success(format_json(sources))

        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            additional_kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": tool.metadata.get_name(),
            }
            if not tool:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Tool {tool_call.tool_name} does not exist",
                        additional_kwargs=additional_kwargs,
                    )
                )
                continue

            try:
                tool_output = tool(**tool_call.tool_kwargs)
                sources.append(tool_output)
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=tool_output.content,
                        additional_kwargs=additional_kwargs,
                    )
                )
            except Exception as e:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Encountered error in tool call: {e}",
                        additional_kwargs=additional_kwargs,
                    )
                )

        async def run_async_code_7cb3b9e4():
            async def run_async_code_94b374ec():
                memory = await ctx.store.get("memory")
                return memory
            memory = asyncio.run(run_async_code_94b374ec())
            logger.success(format_json(memory))
            return memory
        memory = asyncio.run(run_async_code_7cb3b9e4())
        logger.success(format_json(memory))
        for msg in tool_msgs:
            memory.put(msg)

        async def run_async_code_3bf96478():
            await ctx.store.set("sources", sources)
            return 
         = asyncio.run(run_async_code_3bf96478())
        logger.success(format_json())
        async def run_async_code_bbf63658():
            await ctx.store.set("memory", memory)
            return 
         = asyncio.run(run_async_code_bbf63658())
        logger.success(format_json())

        chat_history = memory.get()
        return InputEvent(input=chat_history)

"""
And thats it! Let's explore the workflow we wrote a bit.

`prepare_chat_history()`:
This is our main entry point. It handles adding the user message to memory, and uses the memory to get the latest chat history. It returns an `InputEvent`.

`handle_llm_input()`:
Triggered by an `InputEvent`, it uses the chat history and tools to prompt the llm. If tool calls are found, a `ToolCallEvent` is emitted. Otherwise, we say the workflow is done an emit a `StopEvent`

`handle_tool_calls()`:
Triggered by `ToolCallEvent`, it calls tools with error handling and returns tool outputs. This event triggers a **loop** since it emits an `InputEvent`, which takes us back to `handle_llm_input()`

## Run the Workflow!

**NOTE:** With loops, we need to be mindful of runtime. Here, we set a timeout of 120s.
"""
logger.info("## Run the Workflow!")



def add(x: int, y: int) -> int:
    """Useful function to add two numbers."""
    return x + y


def multiply(x: int, y: int) -> int:
    """Useful function to multiply two numbers."""
    return x * y


tools = [
    FunctionTool.from_defaults(add),
    FunctionTool.from_defaults(multiply),
]

agent = FuncationCallingAgent(
    llm=MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit-mini"), tools=tools, timeout=120, verbose=True
)

async def run_async_code_e75b1b7f():
    async def run_async_code_9a074847():
        ret = await agent.run(input="Hello!")
        return ret
    ret = asyncio.run(run_async_code_9a074847())
    logger.success(format_json(ret))
    return ret
ret = asyncio.run(run_async_code_e75b1b7f())
logger.success(format_json(ret))

logger.debug(ret["response"])

async def run_async_code_8783ece2():
    async def run_async_code_496457ef():
        ret = await agent.run(input="What is (2123 + 2321) * 312?")
        return ret
    ret = asyncio.run(run_async_code_496457ef())
    logger.success(format_json(ret))
    return ret
ret = asyncio.run(run_async_code_8783ece2())
logger.success(format_json(ret))

"""
## Chat History

By default, the workflow is creating a fresh `Context` for each run. This means that the chat history is not preserved between runs. However, we can pass our own `Context` to the workflow to preserve chat history.
"""
logger.info("## Chat History")


ctx = Context(agent)

async def run_async_code_995d19f3():
    async def run_async_code_37139dfa():
        ret = await agent.run(input="Hello! My name is Logan.", ctx=ctx)
        return ret
    ret = asyncio.run(run_async_code_37139dfa())
    logger.success(format_json(ret))
    return ret
ret = asyncio.run(run_async_code_995d19f3())
logger.success(format_json(ret))
logger.debug(ret["response"])

async def run_async_code_2ec36343():
    async def run_async_code_09595dbd():
        ret = await agent.run(input="What is my name?", ctx=ctx)
        return ret
    ret = asyncio.run(run_async_code_09595dbd())
    logger.success(format_json(ret))
    return ret
ret = asyncio.run(run_async_code_2ec36343())
logger.success(format_json(ret))
logger.debug(ret["response"])

"""
## Streaming

Using the `handler` returned from the `.run()` method, we can also access the streaming events.
"""
logger.info("## Streaming")

agent = FuncationCallingAgent(
    llm=MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit-mini"), tools=tools, timeout=120, verbose=False
)

handler = agent.run(input="Hello! Write me a short story about a cat.")

async for event in handler.stream_events():
    if isinstance(event, StreamEvent):
        logger.debug(event.delta, end="", flush=True)

async def run_async_code_78ff5c2b():
    async def run_async_code_2cbcd794():
        response = await handler
        return response
    response = asyncio.run(run_async_code_2cbcd794())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_78ff5c2b())
logger.success(format_json(response))

logger.info("\n\n[DONE]", bright=True)