import asyncio
from jet.transformers.formatters import format_json
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
import os
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPSpanExporter,
)
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    import asyncio
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.workflow import Event
from typing import Any, List
from llama_index.core.agent.react import ReActChatFormatter, ReActOutputParser
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools.types import BaseTool
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from jet.llm.ollama import Ollama
from llama_index.core.tools import FunctionTool
from jet.llm.ollama import Ollama

initialize_ollama_settings()

"""
# Workflow for a ReAct Agent

This notebook walks through setting up a `Workflow` to construct a ReAct agent from (mostly) scratch.

React calling agents work by prompting an LLM to either invoke tools/functions, or return a final response.

Our workflow will be stateful with memory, and will be able to call the LLM to select tools and process incoming user messages.
"""

# !pip install -U llama-index


# os.environ["OPENAI_API_KEY"] = "sk-proj--..."

"""
### [Optional] Set up observability with Llamatrace

Set up tracing to visualize each step in the workflow.
"""

# !pip install "llama-index-core>=0.10.43" "openinference-instrumentation-llama-index>=2" "opentelemetry-proto>=1.12.0" opentelemetry-exporter-otlp opentelemetry-sdk



PHOENIX_API_KEY = "<YOUR-PHOENIX-API-KEY>"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"

span_phoenix_processor = SimpleSpanProcessor(
    HTTPSpanExporter(endpoint="https://app.phoenix.arize.com/v1/traces")
)

tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(span_processor=span_phoenix_processor)

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

"""
Since workflows are async first, this all runs fine in a notebook. If you were running in your own code, you would want to use `asyncio.run()` to start an async event loop if one isn't already running.

```python
async def main():
    <async code>

if __name__ == "__main__":
    asyncio.run(main())
```
"""

"""
## Designing the Workflow

An agent consists of several steps
1. Handling the latest incoming user message, including adding to memory and preparing the chat history
2. Using the chat history and tools to construct a ReAct prompt
3. Calling the llm with the react prompt, and parsing out function/tool calls
4. If no tool calls, we can return
5. If there are tool calls, we need to execute them, and then loop back for a fresh ReAct prompt using the latest tool calls

### The Workflow Events

To handle these steps, we need to define a few events:
1. An event to handle new messages and prepare the chat history
2. An event to prompt the LLM with the react prompt
3. An event to trigger tool calls, if any
4. An event to handle the results of tool calls, if any

The other steps will use the built-in `StartEvent` and `StopEvent` events.

In addition to events, we will also use the global context to store the current react reasoning!
"""



class PrepEvent(Event):
    pass


class InputEvent(Event):
    input: list[ChatMessage]


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class FunctionOutputEvent(Event):
    output: ToolOutput

"""
### The Workflow Itself

With our events defined, we can construct our workflow and steps. 

Note that the workflow automatically validates itself using type annotations, so the type annotations on our steps are very helpful!
"""




class ReActAgent(Workflow):
    def __init__(
        self,
        *args: Any,
        llm: LLM | None = None,
        tools: list[BaseTool] | None = None,
        extra_context: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []

        self.llm = llm or Ollama()

        self.memory = ChatMemoryBuffer.from_defaults(llm=llm)
        self.formatter = ReActChatFormatter(context=extra_context or "")
        self.output_parser = ReActOutputParser()
        self.sources = []

    @step
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> PrepEvent:
        self.sources = []

        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        self.memory.put(user_msg)


        return PrepEvent()

    @step
    async def prepare_chat_history(
        self, ctx: Context, ev: PrepEvent
    ) -> InputEvent:
        chat_history = self.memory.get()
        llm_input = self.formatter.format(
            self.tools, chat_history, current_reasoning=current_reasoning
        )
        return InputEvent(input=llm_input)

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        chat_history = ev.input


        try:
            reasoning_step = self.output_parser.parse(response.message.content)
            (await ctx.get("current_reasoning", default=[])).append(
                reasoning_step
            )
            if reasoning_step.is_done:
                self.memory.put(
                    ChatMessage(
                        role="assistant", content=reasoning_step.response
                    )
                )
                return StopEvent(
                    result={
                        "response": reasoning_step.response,
                        "sources": [*self.sources],
                        "reasoning": await ctx.get(
                            "current_reasoning", default=[]
                        ),
                    }
                )
            elif isinstance(reasoning_step, ActionReasoningStep):
                tool_name = reasoning_step.action
                tool_args = reasoning_step.action_input
                return ToolCallEvent(
                    tool_calls=[
                        ToolSelection(
                            tool_id="fake",
                            tool_name=tool_name,
                            tool_kwargs=tool_args,
                        )
                    ]
                )
        except Exception as e:
            (await ctx.get("current_reasoning", default=[])).append(
                ObservationReasoningStep(
                    observation=f"There was an error in parsing my reasoning: {e}"
                )
            )

        return PrepEvent()

    @step
    async def handle_tool_calls(
        self, ctx: Context, ev: ToolCallEvent
    ) -> PrepEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            if not tool:
                (await ctx.get("current_reasoning", default=[])).append(
                    ObservationReasoningStep(
                        observation=f"Tool {tool_call.tool_name} does not exist"
                    )
                )
                continue

            try:
                tool_output = tool(**tool_call.tool_kwargs)
                self.sources.append(tool_output)
                (await ctx.get("current_reasoning", default=[])).append(
                    ObservationReasoningStep(observation=tool_output.content)
                )
            except Exception as e:
                (await ctx.get("current_reasoning", default=[])).append(
                    ObservationReasoningStep(
                        observation=f"Error calling tool {tool.metadata.get_name()}: {e}"
                    )
                )

        return PrepEvent()

"""
And thats it! Let's explore the workflow we wrote a bit.

`new_user_msg()`:
Adds the user message to memory, and clears the global context to keep track of a fresh string of reasoning.

`prepare_chat_history()`:
Prepares the react prompt, using the chat history, tools, and current reasoning (if any)

`handle_llm_input()`:
Prompts the LLM with our react prompt, and uses some utility functions to parse the output. If there are no tool calls, we can stop and emit a `StopEvent`. Otherwise, we emit a `ToolCallEvent` to handle tool calls. Lastly, if there are no tool calls, and no final response, we simply loop again.

`handle_tool_calls()`:
Safely calls tools with error handling, adding the tool outputs to the current reasoning. Then, by emitting a `PrepEvent`, we loop around for another round of ReAct prompting and parsing.
"""

"""
## Run the Workflow!

**NOTE:** With loops, we need to be mindful of runtime. Here, we set a timeout of 120s.
"""



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

agent = ReActAgent(
    llm=Ollama(model="llama3.1", request_timeout=300.0, context_window=4096), tools=tools, timeout=120, verbose=True
)

async def run_async_code_e75b1b7f():
  ret = await agent.run(input="Hello!")
  return ret

ret = asyncio.run(run_async_code_e75b1b7f())
logger.success(format_json(ret))

print(ret["response"])

async def run_async_code_8783ece2():
  ret = await agent.run(input="What is (2123 + 2321) * 312?")
  return ret

ret = asyncio.run(run_async_code_8783ece2())
logger.success(format_json(ret))

print(ret["response"])

logger.info("\n\n[DONE]", bright=True)