import asyncio
from jet.transformers.formatters import format_json
from autogen_core import AgentId
from autogen_core import EVENT_LOGGER_NAME
from autogen_core import MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from autogen_core.models import CreateResult, UserMessage
from autogen_core.models import UserMessage
from autogen_ext.cache_store.diskcache import DiskCacheStore
from autogen_ext.models.cache import CHAT_CACHE_VALUE_TYPE, ChatCompletionCache
from autogen_ext.models.openai import OllamaChatCompletionClient
from dataclasses import dataclass
from diskcache import Cache
from jet.logger import CustomLogger
from pydantic import BaseModel
from typing import Literal
import asyncio
import logging
import os
import tempfile

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Model Clients

AutoGen provides a suite of built-in model clients for using ChatCompletion API.
All model clients implement the {py:class}`~autogen_core.models.ChatCompletionClient` protocol class.

Currently we support the following built-in model clients:
* {py:class}`~autogen_ext.models.openai.OllamaChatCompletionClient`: for Ollama models and models with Ollama API compatibility (e.g., Gemini).
* {py:class}`~autogen_ext.models.openai.AzureOllamaChatCompletionClient`: for Azure Ollama models.
* {py:class}`~autogen_ext.models.azure.AzureAIChatCompletionClient`: for GitHub models and models hosted on Azure.
* {py:class}`~autogen_ext.models.ollama.OllamaChatCompletionClient` (Experimental): for local models hosted on Ollama.
* {py:class}`~autogen_ext.models.anthropic.AnthropicChatCompletionClient` (Experimental): for models hosted on Anthropic.
* {py:class}`~autogen_ext.models.semantic_kernel.SKChatCompletionAdapter`: adapter for Semantic Kernel AI connectors.

For more information on how to use these model clients, please refer to the documentation of each client.

## Log Model Calls

AutoGen uses standard Python logging module to log events like model calls and responses.
The logger name is {py:attr}`autogen_core.EVENT_LOGGER_NAME`, and the event type is `LLMCall`.
"""
logger.info("# Model Clients")



logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

"""
## Call Model Client

To call a model client, you can use the {py:meth}`~autogen_core.models.ChatCompletionClient.create` method.
This example uses the {py:class}`~autogen_ext.models.openai.OllamaChatCompletionClient` to call an Ollama model.
"""
logger.info("## Call Model Client")


model_client = OllamaChatCompletionClient(
    model="llama3.1", request_timeout=300.0, context_window=4096, temperature=0.3
# )  # assuming OPENAI_API_KEY is set in the environment.

async def run_async_code_75a3b6ab():
    async def run_async_code_4009c609():
        result = await model_client.create([UserMessage(content="What is the capital of France?", source="user")])
        return result
    result = asyncio.run(run_async_code_4009c609())
    logger.success(format_json(result))
    return result
result = asyncio.run(run_async_code_75a3b6ab())
logger.success(format_json(result))
logger.debug(result)

"""
## Streaming Tokens

You can use the {py:meth}`~autogen_core.models.ChatCompletionClient.create_stream` method to create a
chat completion request with streaming token chunks.
"""
logger.info("## Streaming Tokens")


# model_client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096)  # assuming OPENAI_API_KEY is set in the environment.

messages = [
    UserMessage(content="Write a very short story about a dragon.", source="user"),
]

stream = model_client.create_stream(messages=messages)

logger.debug("Streamed responses:")
async for chunk in stream:  # type: ignore
    if isinstance(chunk, str):
        logger.debug(chunk, flush=True, end="")
    else:
        assert isinstance(chunk, CreateResult) and isinstance(chunk.content, str)
        logger.debug("\n\n------------\n")
        logger.debug("The complete response:", flush=True)
        logger.debug(chunk.content, flush=True)

"""
```{note}
The last response in the streaming response is always the final response
of the type {py:class}`~autogen_core.models.CreateResult`.
```

```{note}
The default usage response is to return zero values. To enable usage, 
see {py:meth}`~autogen_ext.models.openai.BaseOllamaChatCompletionClient.create_stream`
for more details.
```

## Structured Output

Structured output can be enabled by setting the `response_format` field in
{py:class}`~autogen_ext.models.openai.OllamaChatCompletionClient` and {py:class}`~autogen_ext.models.openai.AzureOllamaChatCompletionClient` to
as a [Pydantic BaseModel](https://docs.pydantic.dev/latest/concepts/models/) class.

```{note}
Structured output is only available for models that support it. It also
requires the model client to support structured output as well.
Currently, the {py:class}`~autogen_ext.models.openai.OllamaChatCompletionClient`
and {py:class}`~autogen_ext.models.openai.AzureOllamaChatCompletionClient`
support structured output.
```
"""
logger.info("## Structured Output")




class AgentResponse(BaseModel):
    thoughts: str
    response: Literal["happy", "sad", "neutral"]


model_client = OllamaChatCompletionClient(
    model="llama3.1", request_timeout=300.0, context_window=4096,
    response_format=AgentResponse,  # type: ignore
)

messages = [
    UserMessage(content="I am happy.", source="user"),
]
async def run_async_code_fbb22dd6():
    async def run_async_code_8a7414d3():
        response = await model_client.create(messages=messages)
        return response
    response = asyncio.run(run_async_code_8a7414d3())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_fbb22dd6())
logger.success(format_json(response))
assert isinstance(response.content, str)
parsed_response = AgentResponse.model_validate_json(response.content)
logger.debug(parsed_response.thoughts)
logger.debug(parsed_response.response)

async def run_async_code_0349fda4():
    await model_client.close()
    return 
 = asyncio.run(run_async_code_0349fda4())
logger.success(format_json())

"""
You also use the `extra_create_args` parameter in the {py:meth}`~autogen_ext.models.openai.BaseOllamaChatCompletionClient.create` method
to set the `response_format` field so that the structured output can be configured for each request.

## Caching Model Responses

`autogen_ext` implements {py:class}`~autogen_ext.models.cache.ChatCompletionCache` that can wrap any {py:class}`~autogen_core.models.ChatCompletionClient`. Using this wrapper avoids incurring token usage when querying the underlying client with the same prompt multiple times.

{py:class}`~autogen_core.models.ChatCompletionCache` uses a {py:class}`~autogen_core.CacheStore` protocol. We have implemented some useful variants of {py:class}`~autogen_core.CacheStore` including {py:class}`~autogen_ext.cache_store.diskcache.DiskCacheStore` and {py:class}`~autogen_ext.cache_store.redis.RedisStore`.

Here's an example of using `diskcache` for local caching:
"""
logger.info("## Caching Model Responses")






async def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        openai_model_client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096)

        cache_store = DiskCacheStore[CHAT_CACHE_VALUE_TYPE](Cache(tmpdirname))
        cache_client = ChatCompletionCache(openai_model_client, cache_store)

        async def run_async_code_e23dea5e():
            async def run_async_code_ef3d363b():
                response = await cache_client.create([UserMessage(content="Hello, how are you?", source="user")])
                return response
            response = asyncio.run(run_async_code_ef3d363b())
            logger.success(format_json(response))
            return response
        response = asyncio.run(run_async_code_e23dea5e())
        logger.success(format_json(response))
        logger.debug(response)  # Should print response from Ollama
        async def run_async_code_e23dea5e():
            async def run_async_code_ef3d363b():
                response = await cache_client.create([UserMessage(content="Hello, how are you?", source="user")])
                return response
            response = asyncio.run(run_async_code_ef3d363b())
            logger.success(format_json(response))
            return response
        response = asyncio.run(run_async_code_e23dea5e())
        logger.success(format_json(response))
        logger.debug(response)  # Should print cached response

        async def run_async_code_54283016():
            await openai_model_client.close()
            return 
         = asyncio.run(run_async_code_54283016())
        logger.success(format_json())
        async def run_async_code_dba000a8():
            await cache_client.close()
            return 
         = asyncio.run(run_async_code_dba000a8())
        logger.success(format_json())


asyncio.run(main())

"""
Inspecting `cached_client.total_usage()` (or `model_client.total_usage()`) before and after a cached response should yield idential counts.

Note that the caching is sensitive to the exact arguments provided to `cached_client.create` or `cached_client.create_stream`, so changing `tools` or `json_output` arguments might lead to a cache miss.

## Build an Agent with a Model Client

Let's create a simple AI agent that can respond to messages using the ChatCompletion API.
"""
logger.info("## Build an Agent with a Model Client")




@dataclass
class Message:
    content: str


class SimpleAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("A simple agent")
        self._system_messages = [SystemMessage(content="You are a helpful AI assistant.")]
        self._model_client = model_client

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        user_message = UserMessage(content=message.content, source="user")
        async def async_func_21():
            response = await self._model_client.create(
                self._system_messages + [user_message], cancellation_token=ctx.cancellation_token
            )
            return response
        response = asyncio.run(async_func_21())
        logger.success(format_json(response))
        assert isinstance(response.content, str)
        return Message(content=response.content)

"""
The `SimpleAgent` class is a subclass of the
{py:class}`autogen_core.RoutedAgent` class for the convenience of automatically routing messages to the appropriate handlers.
It has a single handler, `handle_user_message`, which handles message from the user. It uses the `ChatCompletionClient` to generate a response to the message.
It then returns the response to the user, following the direct communication model.

```{note}
The `cancellation_token` of the type {py:class}`autogen_core.CancellationToken` is used to cancel
asynchronous operations. It is linked to async calls inside the message handlers
and can be used by the caller to cancel the handlers.
```
"""
logger.info("The `SimpleAgent` class is a subclass of the")


model_client = OllamaChatCompletionClient(
    model="llama3.1",
)

runtime = SingleThreadedAgentRuntime()
await SimpleAgent.register(
    runtime,
    "simple_agent",
    lambda: SimpleAgent(model_client=model_client),
)
runtime.start()
message = Message("Hello, what are some fun things to do in Seattle?")
async def run_async_code_b614784e():
    async def run_async_code_fdadd2c2():
        response = await runtime.send_message(message, AgentId("simple_agent", "default"))
        return response
    response = asyncio.run(run_async_code_fdadd2c2())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_b614784e())
logger.success(format_json(response))
logger.debug(response.content)
async def run_async_code_4aaa8dea():
    await runtime.stop()
    return 
 = asyncio.run(run_async_code_4aaa8dea())
logger.success(format_json())
async def run_async_code_0349fda4():
    await model_client.close()
    return 
 = asyncio.run(run_async_code_0349fda4())
logger.success(format_json())

"""
The above `SimpleAgent` always responds with a fresh context that contains only
the system message and the latest user's message.
We can use model context classes from {py:mod}`autogen_core.model_context`
to make the agent "remember" previous conversations.
See the [Model Context](./model-context.ipynb) page for more details.

## API Keys From Environment Variables

In the examples above, we show that you can provide the API key through the `api_key` argument. Importantly, the Ollama and Azure Ollama clients use the [openai package](https://github.com/openai/openai-python/blob/3f8d8205ae41c389541e125336b0ae0c5e437661/src/openai/__init__.py#L260), which will automatically read an api key from the environment variable if one is not provided.

# - For Ollama, you can set the `OPENAI_API_KEY` environment variable.  
# - For Azure Ollama, you can set the `AZURE_OPENAI_API_KEY` environment variable. 

In addition, for Gemini (Beta), you can set the `GEMINI_API_KEY` environment variable.

This is a good practice to explore, as it avoids including sensitive api keys in your code.
"""
logger.info("## API Keys From Environment Variables")

logger.info("\n\n[DONE]", bright=True)