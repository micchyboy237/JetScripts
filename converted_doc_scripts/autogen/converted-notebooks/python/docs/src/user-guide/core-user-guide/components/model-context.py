import asyncio
from jet.transformers.formatters import format_json
from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import AssistantMessage, ChatCompletionClient, SystemMessage, UserMessage
from dataclasses import dataclass
from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLXAutogenChatLLMAdapter
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
## Model Context

A model context supports storage and retrieval of Chat Completion messages.
It is always used together with a model client to generate LLM-based responses.

For example, {py:mod}`~autogen_core.model_context.BufferedChatCompletionContext`
is a most-recent-used (MRU) context that stores the most recent `buffer_size`
number of messages. This is useful to avoid context overflow in many LLMs.

Let's see an example that uses
{py:mod}`~autogen_core.model_context.BufferedChatCompletionContext`.
"""
logger.info("## Model Context")



@dataclass
class Message:
    content: str

class SimpleAgentWithContext(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A simple agent")
        self._system_messages = [SystemMessage(content="You are a helpful AI assistant.")]
        self._model_client = model_client
        self._model_context = BufferedChatCompletionContext(buffer_size=5)

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        user_message = UserMessage(content=message.content, source="user")
        await self._model_context.add_message(user_message)
        response = await self._model_client.create(
            self._system_messages + (await self._model_context.get_messages()),
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(response.content, str)
        await self._model_context.add_message(AssistantMessage(content=response.content, source=self.metadata["type"]))
        return Message(content=response.content)

"""
Now let's try to ask follow up questions after the first one.
"""
logger.info("Now let's try to ask follow up questions after the first one.")

model_client = MLXAutogenChatLLMAdapter(
    model="qwen3-1.7b-4bit-mini",
)

runtime = SingleThreadedAgentRuntime()
async def async_func_5():
    await SimpleAgentWithContext.register(
        runtime,
        "simple_agent_context",
        lambda: SimpleAgentWithContext(model_client=model_client),
    )
asyncio.run(async_func_5())
async def run_async_code_1e6ac0a6():
    runtime.start()
asyncio.run(run_async_code_1e6ac0a6())
agent_id = AgentId("simple_agent_context", "default")

message = Message("Hello, what are some fun things to do in Seattle?")
logger.debug(f"Question: {message.content}")
async def run_async_code_3f4c141c():
    response = await runtime.send_message(message, agent_id)
    return response
response = asyncio.run(run_async_code_3f4c141c())
logger.success(format_json(response))
logger.debug(f"Response: {response.content}")
logger.debug("-----")

message = Message("What was the first thing you mentioned?")
logger.debug(f"Question: {message.content}")
async def run_async_code_3f4c141c():
    response = await runtime.send_message(message, agent_id)
    return response
response = asyncio.run(run_async_code_3f4c141c())
logger.success(format_json(response))
logger.debug(f"Response: {response.content}")

async def run_async_code_4aaa8dea():
    await runtime.stop()
asyncio.run(run_async_code_4aaa8dea())
async def run_async_code_0349fda4():
    await model_client.close()
asyncio.run(run_async_code_0349fda4())

"""
From the second response, you can see the agent now can recall its own previous responses.
"""
logger.info("From the second response, you can see the agent now can recall its own previous responses.")

logger.info("\n\n[DONE]", bright=True)