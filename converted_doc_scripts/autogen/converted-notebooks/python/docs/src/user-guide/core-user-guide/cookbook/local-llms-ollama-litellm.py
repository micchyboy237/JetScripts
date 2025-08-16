import asyncio
from jet.transformers.formatters import format_json
from autogen_core import (
AgentId,
DefaultTopicId,
MessageContext,
RoutedAgent,
SingleThreadedAgentRuntime,
default_subscription,
message_handler,
)
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import (
AssistantMessage,
ChatCompletionClient,
SystemMessage,
UserMessage,
)
from dataclasses import dataclass
from jet.llm.mlx.autogen_ext.mlx_chat_completion_client import MLXChatCompletionClient
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
# Local LLMs with LiteLLM & Ollama

In this notebook we'll create two agents, Joe and Cathy who like to tell jokes to each other. The agents will use locally running LLMs.

Follow the guide at https://microsoft.github.io/autogen/docs/topics/non-openai-models/local-litellm-ollama/ to understand how to install LiteLLM and Ollama.

We encourage going through the link, but if you're in a hurry and using Linux, run these:  
  
```
curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama3.2:1b

pip install 'litellm[proxy]'
litellm --model ollama/llama3.2:1b
```  

This will run the proxy server and it will be available at 'http://0.0.0.0:4000/'.

To get started, let's import some classes.
"""
logger.info("# Local LLMs with LiteLLM & Ollama")



"""
Set up out local LLM model client.
"""
logger.info("Set up out local LLM model client.")

def get_model_client() -> MLXChatCompletionClient:  # type: ignore
    "Mimic MLX API using Local LLM Server."
    return MLXChatCompletionClient(
        model="llama3.2:1b",
        api_key="NotRequiredSinceWeAreLocal",
        base_url="http://0.0.0.0:4000",
        model_capabilities={
            "json_output": False,
            "vision": False,
            "function_calling": True,
        },
    )

"""
Define a simple message class
"""
logger.info("Define a simple message class")

@dataclass
class Message:
    content: str

"""
Now, the Agent.

We define the role of the Agent using the `SystemMessage` and set up a condition for termination.
"""
logger.info("Now, the Agent.")

@default_subscription
class Assistant(RoutedAgent):
    def __init__(self, name: str, model_client: ChatCompletionClient) -> None:
        super().__init__("An assistant agent.")
        self._model_client = model_client
        self.name = name
        self.count = 0
        self._system_messages = [
            SystemMessage(
                content=f"Your name is {name} and you are a part of a duo of comedians."
                "You laugh when you find the joke funny, else reply 'I need to go now'.",
            )
        ]
        self._model_context = BufferedChatCompletionContext(buffer_size=5)

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        self.count += 1
        async def run_async_code_37101ad6():
            await self._model_context.add_message(UserMessage(content=message.content, source="user"))
            return 
         = asyncio.run(run_async_code_37101ad6())
        logger.success(format_json())
        async def run_async_code_1c23b3c9():
            async def run_async_code_458a185b():
                result = await self._model_client.create(self._system_messages + await self._model_context.get_messages())
                return result
            result = asyncio.run(run_async_code_458a185b())
            logger.success(format_json(result))
            return result
        result = asyncio.run(run_async_code_1c23b3c9())
        logger.success(format_json(result))

        logger.debug(f"\n{self.name}: {message.content}")

        if "I need to go".lower() in message.content.lower() or self.count > 2:
            return

        async def run_async_code_ddaf37cd():
            await self._model_context.add_message(AssistantMessage(content=result.content, source="assistant"))  # type: ignore
            return 
         = asyncio.run(run_async_code_ddaf37cd())
        logger.success(format_json())
        async def run_async_code_ac73baad():
            await self.publish_message(Message(content=result.content), DefaultTopicId())  # type: ignore
            return 
         = asyncio.run(run_async_code_ac73baad())
        logger.success(format_json())

"""
Set up the agents.
"""
logger.info("Set up the agents.")

runtime = SingleThreadedAgentRuntime()

model_client = get_model_client()

async def async_func_4():
    cathy = await Assistant.register(
        runtime,
        "cathy",
        lambda: Assistant(name="Cathy", model_client=model_client),
    )
    return cathy
cathy = asyncio.run(async_func_4())
logger.success(format_json(cathy))

async def async_func_10():
    joe = await Assistant.register(
        runtime,
        "joe",
        lambda: Assistant(name="Joe", model_client=model_client),
    )
    return joe
joe = asyncio.run(async_func_10())
logger.success(format_json(joe))

"""
Let's run everything!
"""
logger.info("Let's run everything!")

runtime.start()
await runtime.send_message(
    Message("Joe, tell me a joke."),
    recipient=AgentId(joe, "default"),
    sender=AgentId(cathy, "default"),
)
async def run_async_code_b7ca34d4():
    await runtime.stop_when_idle()
    return 
 = asyncio.run(run_async_code_b7ca34d4())
logger.success(format_json())

async def run_async_code_0349fda4():
    await model_client.close()
    return 
 = asyncio.run(run_async_code_0349fda4())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)