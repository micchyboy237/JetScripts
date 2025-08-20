import asyncio
from jet.transformers.formatters import format_json
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler
from autogen_core import SingleThreadedAgentRuntime
from dataclasses import dataclass
from jet.logger import CustomLogger
from openai import AsyncAssistantEventHandler, AsyncClient
from openai.types.beta.thread import ToolResources, ToolResourcesFileSearch
from openai.types.beta.threads import Message, Text, TextDelta
from openai.types.beta.threads.runs import RunStep, RunStepDelta
from typing import Any, Callable, List
from typing_extensions import override
import aiofiles
import asyncio
import logging
import openai
import os
import requests
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# MLX Assistant Agent

[Open AI Assistant](https://platform.openai.com/docs/assistants/overview) 
and [Azure MLX Assistant](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/assistant)
are server-side APIs for building
agents.
They can be used to build agents in AutoGen. This cookbook demonstrates how to
to use MLX Assistant to create an agent that can run code and Q&A over document.

## Message Protocol

First, we need to specify the message protocol for the agent backed by 
MLX Assistant. The message protocol defines the structure of messages
handled and published by the agent. 
For illustration, we define a simple
message protocol of 4 message types: `Message`, `Reset`, `UploadForCodeInterpreter` and `UploadForFileSearch`.
"""
logger.info("# MLX Assistant Agent")



@dataclass
class TextMessage:
    content: str
    source: str


@dataclass
class Reset:
    pass


@dataclass
class UploadForCodeInterpreter:
    file_path: str


@dataclass
class UploadForFileSearch:
    file_path: str
    vector_store_id: str

"""
The `TextMessage` message type is used to communicate with the agent. It has a
`content` field that contains the message content, and a `source` field
for the sender. The `Reset` message type is a control message that resets
the memory of the agent. It has no fields. This is useful when we need to
start a new conversation with the agent.

The `UploadForCodeInterpreter` message type is used to upload data files
for the code interpreter and `UploadForFileSearch` message type is used to upload
documents for file search. Both message types have a `file_path` field that contains
the local path to the file to be uploaded.

## Defining the Agent

Next, we define the agent class.
The agent class constructor has the following arguments: `description`,
`client`, `assistant_id`, `thread_id`, and `assistant_event_handler_factory`.
The `client` argument is the MLX async client object, and the
`assistant_event_handler_factory` is for creating an assistant event handler
to handle MLX Assistant events.
This can be used to create streaming output from the assistant.

The agent class has the following message handlers:
- `handle_message`: Handles the `TextMessage` message type, and sends back the
  response from the assistant.
- `handle_reset`: Handles the `Reset` message type, and resets the memory
    of the assistant agent.
- `handle_upload_for_code_interpreter`: Handles the `UploadForCodeInterpreter`
  message type, and uploads the file to the code interpreter.
- `handle_upload_for_file_search`: Handles the `UploadForFileSearch`
    message type, and uploads the document to the file search.


The memory of the assistant is stored inside a thread, which is kept in the
server side. The thread is referenced by the `thread_id` argument.
"""
logger.info("## Defining the Agent")




class MLXAssistantAgent(RoutedAgent):
    """An agent implementation that uses the MLX Assistant API to generate
    responses.

    Args:
        description (str): The description of the agent.
        client (openai.AsyncClient): The client to use for the MLX API.
        assistant_id (str): The assistant ID to use for the MLX API.
        thread_id (str): The thread ID to use for the MLX API.
        assistant_event_handler_factory (Callable[[], AsyncAssistantEventHandler], optional):
            A factory function to create an async assistant event handler. Defaults to None.
            If provided, the agent will use the streaming mode with the event handler.
            If not provided, the agent will use the blocking mode to generate responses.
    """

    def __init__(
        self,
        description: str,
        client: AsyncClient,
        assistant_id: str,
        thread_id: str,
        assistant_event_handler_factory: Callable[[], AsyncAssistantEventHandler],
    ) -> None:
        super().__init__(description)
        self._client = client
        self._assistant_id = assistant_id
        self._thread_id = thread_id
        self._assistant_event_handler_factory = assistant_event_handler_factory

    @message_handler
    async def handle_message(self, message: TextMessage, ctx: MessageContext) -> TextMessage:
        """Handle a message. This method adds the message to the thread and publishes a response."""
        await ctx.cancellation_token.link_future(
            asyncio.ensure_future(
                self._client.beta.threads.messages.create(
                    thread_id=self._thread_id,
                    content=message.content,
                    role="user",
                    metadata={"sender": message.source},
                )
            )
        )
        async with self._client.beta.threads.runs.stream(
            thread_id=self._thread_id,
            assistant_id=self._assistant_id,
            event_handler=self._assistant_event_handler_factory(),
        ) as stream:
            await ctx.cancellation_token.link_future(asyncio.ensure_future(stream.until_done()))

        async def async_func_59():
            messages = await ctx.cancellation_token.link_future(
                asyncio.ensure_future(self._client.beta.threads.messages.list(self._thread_id, order="desc", limit=1))
            )
            return messages
        messages = asyncio.run(async_func_59())
        logger.success(format_json(messages))
        last_message_content = messages.data[0].content

        text_content = [content for content in last_message_content if content.type == "text"]
        if not text_content:
            raise ValueError(f"Expected text content in the last message: {last_message_content}")

        return TextMessage(content=text_content[0].text.value, source=self.metadata["type"])

    @message_handler()
    async def on_reset(self, message: Reset, ctx: MessageContext) -> None:
        """Handle a reset message. This method deletes all messages in the thread."""
        all_msgs: List[str] = []
        while True:
            if not all_msgs:
                msgs = await ctx.cancellation_token.link_future(
                    asyncio.ensure_future(self._client.beta.threads.messages.list(self._thread_id))
                )
            else:
                msgs = await ctx.cancellation_token.link_future(
                    asyncio.ensure_future(self._client.beta.threads.messages.list(self._thread_id, after=all_msgs[-1]))
                )
            for msg in msgs.data:
                all_msgs.append(msg.id)
            if not msgs.has_next_page():
                break
        for msg_id in all_msgs:
            status = await ctx.cancellation_token.link_future(
                asyncio.ensure_future(
                    self._client.beta.threads.messages.delete(message_id=msg_id, thread_id=self._thread_id)
                )
            )
            assert status.deleted is True

    @message_handler()
    async def on_upload_for_code_interpreter(self, message: UploadForCodeInterpreter, ctx: MessageContext) -> None:
        """Handle an upload for code interpreter. This method uploads a file and updates the thread with the file."""
        async with aiofiles.open(message.file_path, mode="rb") as f:
            file_content = await ctx.cancellation_token.link_future(asyncio.ensure_future(f.read()))
        file_name = os.path.basename(message.file_path)
        file = await ctx.cancellation_token.link_future(
            asyncio.ensure_future(self._client.files.create(file=(file_name, file_content), purpose="assistants"))
        )
        thread = await ctx.cancellation_token.link_future(
            asyncio.ensure_future(self._client.beta.threads.retrieve(thread_id=self._thread_id))
        )
        tool_resources: ToolResources = thread.tool_resources if thread.tool_resources else ToolResources()
        assert tool_resources.code_interpreter is not None
        if tool_resources.code_interpreter.file_ids:
            file_ids = tool_resources.code_interpreter.file_ids
        else:
            file_ids = [file.id]
        await ctx.cancellation_token.link_future(
            asyncio.ensure_future(
                self._client.beta.threads.update(
                    thread_id=self._thread_id,
                    tool_resources={
                        "code_interpreter": {"file_ids": file_ids},
                    },
                )
            )
        )

    @message_handler()
    async def on_upload_for_file_search(self, message: UploadForFileSearch, ctx: MessageContext) -> None:
        """Handle an upload for file search. This method uploads a file and updates the vector store."""
        async with aiofiles.open(message.file_path, mode="rb") as file:
            file_content = await ctx.cancellation_token.link_future(asyncio.ensure_future(file.read()))
        file_name = os.path.basename(message.file_path)
        await ctx.cancellation_token.link_future(
            asyncio.ensure_future(
                self._client.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=message.vector_store_id,
                    files=[(file_name, file_content)],
                )
            )
        )

"""
The agent class is a thin wrapper around the MLX Assistant API to implement
the message protocol. More features, such as multi-modal message handling,
can be added by extending the message protocol.

## Assistant Event Handler

The assistant event handler provides call-backs for handling Assistant API
specific events. This is useful for handling streaming output from the assistant
and further user interface integration.
"""
logger.info("## Assistant Event Handler")



class EventHandler(AsyncAssistantEventHandler):
    @override
    async def on_text_delta(self, delta: TextDelta, snapshot: Text) -> None:
        logger.debug(delta.value, end="", flush=True)

    @override
    async def on_run_step_created(self, run_step: RunStep) -> None:
        details = run_step.step_details
        if details.type == "tool_calls":
            for tool in details.tool_calls:
                if tool.type == "code_interpreter":
                    logger.debug("\nGenerating code to interpret:\n\n```python")

    @override
    async def on_run_step_done(self, run_step: RunStep) -> None:
        details = run_step.step_details
        if details.type == "tool_calls":
            for tool in details.tool_calls:
                if tool.type == "code_interpreter":
                    logger.debug("\n```\nExecuting code...")

    @override
    async def on_run_step_delta(self, delta: RunStepDelta, snapshot: RunStep) -> None:
        details = delta.step_details
        if details is not None and details.type == "tool_calls":
            for tool in details.tool_calls or []:
                if tool.type == "code_interpreter" and tool.code_interpreter and tool.code_interpreter.input:
                    logger.debug(tool.code_interpreter.input, end="", flush=True)

    @override
    async def on_message_created(self, message: Message) -> None:
        logger.debug(f"{'-'*80}\nAssistant:\n")

    @override
    async def on_message_done(self, message: Message) -> None:
        if not message.content:
            return
        content = message.content[0]
        if not content.type == "text":
            return
        text_content = content.text
        annotations = text_content.annotations
        citations: List[str] = []
        for index, annotation in enumerate(annotations):
            text_content.value = text_content.value.replace(annotation.text, f"[{index}]")
            if file_citation := getattr(annotation, "file_citation", None):
                client = AsyncClient()
                cited_file = await client.files.retrieve(file_citation.file_id)
                citations.append(f"[{index}] {cited_file.filename}")
        if citations:
            logger.debug("\n".join(citations))

"""
## Using the Agent

First we need to use the `openai` client to create the actual assistant,
thread, and vector store. Our AutoGen agent will be using these.
"""
logger.info("## Using the Agent")


oai_assistant = openai.beta.assistants.create(
    model="qwen3-1.7b-4bit-mini",
    description="An AI assistant that helps with everyday tasks.",
    instructions="Help the user with their task.",
    tools=[{"type": "code_interpreter"}, {"type": "file_search"}],
)

vector_store = openai.vector_stores.create()

thread = openai.beta.threads.create(
    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
)

"""
Then, we create a runtime, and register an agent factory function for this 
agent with the runtime.
"""
logger.info("Then, we create a runtime, and register an agent factory function for this")


runtime = SingleThreadedAgentRuntime()
async def async_func_3():
    await MLXAssistantAgent.register(
        runtime,
        "assistant",
        lambda: MLXAssistantAgent(
            description="MLX Assistant Agent",
            client=openai.AsyncClient(),
            assistant_id=oai_assistant.id,
            thread_id=thread.id,
            assistant_event_handler_factory=lambda: EventHandler(),
        ),
    )
asyncio.run(async_func_3())
agent = AgentId("assistant", "default")

"""
Let's turn on logging to see what's happening under the hood.
"""
logger.info("Let's turn on logging to see what's happening under the hood.")


logging.basicConfig(level=logging.WARNING)
logging.getLogger("autogen_core").setLevel(logging.DEBUG)

"""
Let's send a greeting message to the agent, and see the response streamed back.
"""
logger.info("Let's send a greeting message to the agent, and see the response streamed back.")

async def run_async_code_1e6ac0a6():
    runtime.start()
asyncio.run(run_async_code_1e6ac0a6())
async def run_async_code_6407d14a():
    await runtime.send_message(TextMessage(content="Hello, how are you today!", source="user"), agent)
asyncio.run(run_async_code_6407d14a())
async def run_async_code_b7ca34d4():
    await runtime.stop_when_idle()
asyncio.run(run_async_code_b7ca34d4())

"""
## Assistant with Code Interpreter

Let's ask some math question to the agent, and see it uses the code interpreter
to answer the question.
"""
logger.info("## Assistant with Code Interpreter")

async def run_async_code_1e6ac0a6():
    runtime.start()
asyncio.run(run_async_code_1e6ac0a6())
async def run_async_code_30a4d357():
    await runtime.send_message(TextMessage(content="What is 1332322 x 123212?", source="user"), agent)
asyncio.run(run_async_code_30a4d357())
async def run_async_code_b7ca34d4():
    await runtime.stop_when_idle()
asyncio.run(run_async_code_b7ca34d4())

"""
Let's get some data from Seattle Open Data portal. We will be using the
[City of Seattle Wage Data](https://data.seattle.gov/City-Business/City-of-Seattle-Wage-Data/2khk-5ukd/).
Let's download it first.
"""
logger.info("Let's get some data from Seattle Open Data portal. We will be using the")


response = requests.get("https://data.seattle.gov/resource/2khk-5ukd.csv")
with open("seattle_city_wages.csv", "wb") as file:
    file.write(response.content)

"""
Let's send the file to the agent using an `UploadForCodeInterpreter` message.
"""
logger.info("Let's send the file to the agent using an `UploadForCodeInterpreter` message.")

async def run_async_code_1e6ac0a6():
    runtime.start()
asyncio.run(run_async_code_1e6ac0a6())
async def run_async_code_5fd06a41():
    await runtime.send_message(UploadForCodeInterpreter(file_path="seattle_city_wages.csv"), agent)
asyncio.run(run_async_code_5fd06a41())
async def run_async_code_b7ca34d4():
    await runtime.stop_when_idle()
asyncio.run(run_async_code_b7ca34d4())

"""
We can now ask some questions about the data to the agent.
"""
logger.info("We can now ask some questions about the data to the agent.")

async def run_async_code_1e6ac0a6():
    runtime.start()
asyncio.run(run_async_code_1e6ac0a6())
async def run_async_code_36dcec0b():
    await runtime.send_message(TextMessage(content="Take a look at the uploaded CSV file.", source="user"), agent)
asyncio.run(run_async_code_36dcec0b())
async def run_async_code_b7ca34d4():
    await runtime.stop_when_idle()
asyncio.run(run_async_code_b7ca34d4())

async def run_async_code_1e6ac0a6():
    runtime.start()
asyncio.run(run_async_code_1e6ac0a6())
async def run_async_code_f819bab5():
    await runtime.send_message(TextMessage(content="What are the top-10 salaries?", source="user"), agent)
asyncio.run(run_async_code_f819bab5())
async def run_async_code_b7ca34d4():
    await runtime.stop_when_idle()
asyncio.run(run_async_code_b7ca34d4())

"""
## Assistant with File Search

Let's try the Q&A over document feature. We first download Wikipedia page
on the Third Anglo-Afghan War.
"""
logger.info("## Assistant with File Search")

response = requests.get("https://en.wikipedia.org/wiki/Third_Anglo-Afghan_War")
with open("third_anglo_afghan_war.html", "wb") as file:
    file.write(response.content)

"""
Send the file to the agent using an `UploadForFileSearch` message.
"""
logger.info("Send the file to the agent using an `UploadForFileSearch` message.")

async def run_async_code_1e6ac0a6():
    runtime.start()
asyncio.run(run_async_code_1e6ac0a6())
async def async_func_1():
    await runtime.send_message(
        UploadForFileSearch(file_path="third_anglo_afghan_war.html", vector_store_id=vector_store.id), agent
    )
asyncio.run(async_func_1())
async def run_async_code_b7ca34d4():
    await runtime.stop_when_idle()
asyncio.run(run_async_code_b7ca34d4())

"""
Let's ask some questions about the document to the agent. Before asking,
we reset the agent memory to start a new conversation.
"""
logger.info("Let's ask some questions about the document to the agent. Before asking,")

async def run_async_code_1e6ac0a6():
    runtime.start()
asyncio.run(run_async_code_1e6ac0a6())
async def run_async_code_4488944a():
    await runtime.send_message(Reset(), agent)
asyncio.run(run_async_code_4488944a())
async def async_func_2():
    await runtime.send_message(
        TextMessage(
            content="When and where was the treaty of Rawalpindi signed? Answer using the document provided.", source="user"
        ),
        agent,
    )
asyncio.run(async_func_2())
async def run_async_code_b7ca34d4():
    await runtime.stop_when_idle()
asyncio.run(run_async_code_b7ca34d4())

"""
That's it! We have successfully built an agent backed by MLX Assistant.
"""
logger.info("That's it! We have successfully built an agent backed by MLX Assistant.")

logger.info("\n\n[DONE]", bright=True)