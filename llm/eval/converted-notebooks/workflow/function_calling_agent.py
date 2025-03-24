from llama_index.core.tools import FunctionTool
from typing import Any, List
from jet.llm.ollama.base import Ollama
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step
from llama_index.core.tools.types import BaseTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.workflow import Event
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.llms import ChatMessage
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPSpanExporter,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk import trace as trace_sdk
import os
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# Workflow for a Function Calling Agent
#
# This notebook walks through setting up a `Workflow` to construct a function calling agent from scratch.
#
# Function calling agents work by using an LLM that supports tools/functions in its API (Ollama, Ollama, Anthropic, etc.) to call functions an use tools.
#
# Our workflow will be stateful with memory, and will be able to call the LLM to select tools and process incoming user messages.

# !pip install -U llama-index


# os.environ["OPENAI_API_KEY"] = "sk-proj-..."

# [Optional] Set up observability with Llamatrace
#
# Set up tracing to visualize each step in the workflow.

# !pip install "llama-index-core>=0.10.43" "openinference-instrumentation-llama-index>=2.2.2" "opentelemetry-proto>=1.12.0" opentelemetry-exporter-otlp opentelemetry-sdk


PHOENIX_API_KEY = "<YOUR-PHOENIX-API-KEY>"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"

span_phoenix_processor = SimpleSpanProcessor(
    HTTPSpanExporter(endpoint="https://app.phoenix.arize.com/v1/traces")
)

tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(span_processor=span_phoenix_processor)

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

# Since workflows are async first, this all runs fine in a notebook. If you were running in your own code, you would want to use `asyncio.run()` to start an async event loop if one isn't already running.
#
# ```python
# async def main():
#     <async code>
#
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
# ```

# Designing the Workflow
#
# An agent consists of several steps
# 1. Handling the latest incoming user message, including adding to memory and getting the latest chat history
# 2. Calling the LLM with tools + chat history
# 3. Parsing out tool calls (if any)
# 4. If there are tool calls, call them, and loop until there are none
# 5. When there is no tool calls, return the LLM response
#
# The Workflow Events
#
# To handle these steps, we need to define a few events:
# 1. An event to handle new messages and prepare the chat history
# 2. An event to trigger tool calls
# 3. An event to handle the results of tool calls
#
# The other steps will use the built-in `StartEvent` and `StopEvent` events.


class InputEvent(Event):
    input: list[ChatMessage]


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class FunctionOutputEvent(Event):
    output: ToolOutput

# The Workflow Itself
#
# With our events defined, we can construct our workflow and steps.
#
# Note that the workflow automatically validates itself using type annotations, so the type annotations on our steps are very helpful!


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

        self.llm = llm or Ollama()
        assert self.llm.metadata.is_function_calling_model

        self.memory = ChatMemoryBuffer.from_defaults(llm=llm)
        self.sources = []

    @step
    async def prepare_chat_history(self, ev: StartEvent) -> InputEvent:
        self.sources = []

        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        self.memory.put(user_msg)

        chat_history = self.memory.get()
        return InputEvent(input=chat_history)

    @step
    async def handle_llm_input(
        self, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        chat_history = ev.input

        response = self.llm.chat_with_tools(
            self.tools, chat_history=chat_history
        )
        self.memory.put(response.message)

        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )

        if not tool_calls:
            return StopEvent(
                result={"response": response, "sources": [*self.sources]}
            )
        else:
            return ToolCallEvent(tool_calls=tool_calls)

    @step
    async def handle_tool_calls(self, ev: ToolCallEvent) -> InputEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        tool_msgs = []

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
                self.sources.append(tool_output)
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

        for msg in tool_msgs:
            self.memory.put(msg)

        chat_history = self.memory.get()
        return InputEvent(input=chat_history)

# And thats it! Let's explore the workflow we wrote a bit.
#
# `prepare_chat_history()`:
# This is our main entry point. It handles adding the user message to memory, and uses the memory to get the latest chat history. It returns an `InputEvent`.
#
# `handle_llm_input()`:
# Triggered by an `InputEvent`, it uses the chat history and tools to prompt the llm. If tool calls are found, a `ToolCallEvent` is emitted. Otherwise, we say the workflow is done an emit a `StopEvent`
#
# `handle_tool_calls()`:
# Triggered by `ToolCallEvent`, it calls tools with error handling and returns tool outputs. This event triggers a **loop** since it emits an `InputEvent`, which takes us back to `handle_llm_input()`

# Run the Workflow!
#
# **NOTE:** With loops, we need to be mindful of runtime. Here, we set a timeout of 120s.


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
    llm=Ollama(model="llama3.1", request_timeout=300.0, context_window=4096), tools=tools, timeout=120, verbose=True
)

ret = await agent.run(input="Hello!")

print(ret["response"])

ret = await agent.run(input="What is (2123 + 2321) * 312?")

print(ret["response"])

logger.info("\n\n[DONE]", bright=True)
