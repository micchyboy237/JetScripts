async def main():
    from jet.transformers.formatters import format_json
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core.agent.react import ReActChatFormatter, ReActOutputParser
    from llama_index.core.agent.react.types import (
        ActionReasoningStep,
        ObservationReasoningStep,
    )
    from llama_index.core.llms import ChatMessage
    from llama_index.core.llms.llm import LLM
    from llama_index.core.memory import ChatMemoryBuffer
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
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HTTPSpanExporter,
    )
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
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

    """
    # Workflow for a ReAct Agent
    
    This notebook walks through setting up a `Workflow` to construct a ReAct agent from (mostly) scratch.
    
    React calling agents work by prompting an LLM to either invoke tools/functions, or return a final response.
    
    Our workflow will be stateful with memory, and will be able to call the LLM to select tools and process incoming user messages.
    """
    logger.info("# Workflow for a ReAct Agent")

    # !pip install -U llama-index

    # os.environ["OPENAI_API_KEY"] = "sk-proj-..."

    """
    ### [Optional] Set up observability with Llamatrace
    
    Set up tracing to visualize each step in the workflow.
    """
    logger.info("### [Optional] Set up observability with Llamatrace")

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
    2. An event to stream the LLM response
    3. An event to prompt the LLM with the react prompt
    4. An event to trigger tool calls, if any
    5. An event to handle the results of tool calls, if any
    
    The other steps will use the built-in `StartEvent` and `StopEvent` events.
    
    In addition to events, we will also use the global context to store the current react reasoning!
    """
    logger.info("## Designing the Workflow")

    class PrepEvent(Event):
        pass

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
            self.llm = llm or OllamaFunctionCalling()
            self.formatter = ReActChatFormatter.from_defaults(
                context=extra_context or ""
            )
            self.output_parser = ReActOutputParser()

        @step
        async def new_user_msg(self, ctx: Context, ev: StartEvent) -> PrepEvent:
            await ctx.store.set("sources", [])

            memory = await ctx.store.get("memory", default=None)
            logger.success(format_json(memory))
            if not memory:
                memory = ChatMemoryBuffer.from_defaults(llm=self.llm)

            user_input = ev.input
            user_msg = ChatMessage(role="user", content=user_input)
            memory.put(user_msg)

            await ctx.store.set("current_reasoning", [])

            await ctx.store.set("memory", memory)

            return PrepEvent()

        @step
        async def prepare_chat_history(
            self, ctx: Context, ev: PrepEvent
        ) -> InputEvent:
            memory = await ctx.store.get("memory")
            logger.success(format_json(memory))
            chat_history = memory.get()
            current_reasoning = await ctx.store.get(
                "current_reasoning", default=[]
            )
            logger.success(format_json(current_reasoning))

            llm_input = self.formatter.format(
                self.tools, chat_history, current_reasoning=current_reasoning
            )
            return InputEvent(input=llm_input)

        @step
        async def handle_llm_input(
            self, ctx: Context, ev: InputEvent
        ) -> ToolCallEvent | StopEvent:
            chat_history = ev.input
            current_reasoning = await ctx.store.get(
                "current_reasoning", default=[]
            )
            logger.success(format_json(current_reasoning))
            memory = await ctx.store.get("memory")
            logger.success(format_json(memory))

            response_gen = self.llm.stream_chat(chat_history)
            logger.success(format_json(response_gen))
            async for response in response_gen:
                ctx.write_event_to_stream(
                    StreamEvent(delta=response.delta or ""))

            try:
                reasoning_step = self.output_parser.parse(
                    response.message.content)
                current_reasoning.append(reasoning_step)

                if reasoning_step.is_done:
                    memory.put(
                        ChatMessage(
                            role="assistant", content=reasoning_step.response
                        )
                    )
                    await ctx.store.set("memory", memory)
                    await ctx.store.set("current_reasoning", current_reasoning)

                    sources = await ctx.store.get("sources", default=[])
                    logger.success(format_json(sources))

                    return StopEvent(
                        result={
                            "response": reasoning_step.response,
                            "sources": [sources],
                            "reasoning": current_reasoning,
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
                current_reasoning.append(
                    ObservationReasoningStep(
                        observation=f"There was an error in parsing my reasoning: {e}"
                    )
                )
                await ctx.store.set("current_reasoning", current_reasoning)

            return PrepEvent()

        @step
        async def handle_tool_calls(
            self, ctx: Context, ev: ToolCallEvent
        ) -> PrepEvent:
            tool_calls = ev.tool_calls
            tools_by_name = {
                tool.metadata.get_name(): tool for tool in self.tools}
            current_reasoning = await ctx.store.get(
                "current_reasoning", default=[]
            )
            logger.success(format_json(current_reasoning))
            sources = await ctx.store.get("sources", default=[])
            logger.success(format_json(sources))

            for tool_call in tool_calls:
                tool = tools_by_name.get(tool_call.tool_name)
                if not tool:
                    current_reasoning.append(
                        ObservationReasoningStep(
                            observation=f"Tool {tool_call.tool_name} does not exist"
                        )
                    )
                    continue

                try:
                    tool_output = tool(**tool_call.tool_kwargs)
                    sources.append(tool_output)
                    current_reasoning.append(
                        ObservationReasoningStep(
                            observation=tool_output.content)
                    )
                except Exception as e:
                    current_reasoning.append(
                        ObservationReasoningStep(
                            observation=f"Error calling tool {tool.metadata.get_name()}: {e}"
                        )
                    )

            await ctx.store.set("sources", sources)
            await ctx.store.set("current_reasoning", current_reasoning)

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

    agent = ReActAgent(
        llm=OllamaFunctionCalling(model="llama3.2"), tools=tools, timeout=120, verbose=True
    )

    ret = await agent.run(input="Hello!")
    logger.success(format_json(ret))

    logger.debug(ret["response"])

    ret = await agent.run(input="What is (2123 + 2321) * 312?")
    logger.success(format_json(ret))

    logger.debug(ret["response"])

    """
    ## Chat History
    
    By default, the workflow is creating a fresh `Context` for each run. This means that the chat history is not preserved between runs. However, we can pass our own `Context` to the workflow to preserve chat history.
    """
    logger.info("## Chat History")

    ctx = Context(agent)

    ret = await agent.run(input="Hello! My name is Logan", ctx=ctx)
    logger.success(format_json(ret))
    logger.debug(ret["response"])

    ret = await agent.run(input="What is my name?", ctx=ctx)
    logger.success(format_json(ret))
    logger.debug(ret["response"])

    """
    ## Streaming
    
    We can also access the streaming response from the LLM, using the `handler` object returned from the `.run()` method.
    """
    logger.info("## Streaming")

    agent = ReActAgent(
        llm=OllamaFunctionCalling(model="llama3.2"), tools=tools, timeout=120, verbose=False
    )

    handler = agent.run(input="Hello! Tell me a joke.")

    async for event in handler.stream_events():
        if isinstance(event, StreamEvent):
            logger.debug(event.delta, end="", flush=True)

    response = await handler
    logger.success(format_json(response))

    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())
