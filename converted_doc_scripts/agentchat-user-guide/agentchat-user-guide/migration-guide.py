import asyncio
from jet.transformers.formatters import format_json
from autogen.agentchat import AssistantAgent
from autogen.agentchat import AssistantAgent, GroupChat, GroupChatManager
from autogen.agentchat import AssistantAgent, UserProxyAgent
from autogen.agentchat import AssistantAgent, UserProxyAgent, register_function
from autogen.agentchat import ConversableAgent
from autogen.agentchat import UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor
from autogen.oai import OllamaWrapper
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.base import Response
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.messages import (
BaseAgentEvent,
BaseChatMessage,
HandoffMessage,
MultiModalMessage,
StopMessage,
TextMessage,
ToolCallExecutionEvent,
ToolCallRequestEvent,
ToolCallSummaryMessage,
)
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.messages import MultiModalMessage
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_core import CancellationToken, Image
from autogen_core import FunctionCall, Image
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import ChatCompletionClient
from autogen_core.models import FunctionExecutionResult
from autogen_core.models import UserMessage
from autogen_ext.cache_store.diskcache import DiskCacheStore
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.cache import ChatCompletionCache, CHAT_CACHE_VALUE_TYPE
from autogen_ext.models.openai import AzureOllamaChatCompletionClient
from autogen_ext.models.openai import OllamaChatCompletionClient
from diskcache import Cache
from jet.logger import CustomLogger
from pathlib import Path
from typing import Any, Dict, List, Literal
from typing import Any, Dict, List, Optional, Tuple, Union
from typing import Sequence
import asyncio
import json
import os
import tempfile

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


"""
# Migration Guide for v0.2 to v0.4

This is a migration guide for users of the `v0.2.*` versions of `autogen-agentchat`
to the `v0.4` version, which introduces a new set of APIs and features.
The `v0.4` version contains breaking changes. Please read this guide carefully.
We still maintain the `v0.2` version in the `0.2` branch; however,
we highly recommend you upgrade to the `v0.4` version.
"""
logger.info("# Migration Guide for v0.2 to v0.4")

We no longer have admin access to the `pyautogen` PyPI package, and
the releases from that package are no longer from Microsoft since version 0.2.34.
To continue use the `v0.2` version of AutoGen, install it using `autogen-agentchat~=0.2`.
Please read our [clarification statement](https://github.com/microsoft/autogen/discussions/4217) regarding forks.

"""
## What is `v0.4`?

Since the release of AutoGen in 2023, we have intensively listened to our community and users from small startups and large enterprises, gathering much feedback. Based on that feedback, we built AutoGen `v0.4`, a from-the-ground-up rewrite adopting an asynchronous, event-driven architecture to address issues such as observability, flexibility, interactive control, and scale.

The `v0.4` API is layered:
the [Core API](../core-user-guide/index.md) is the foundation layer offering a scalable, event-driven actor framework for creating agentic workflows;
the [AgentChat API](./index.md) is built on Core, offering a task-driven, high-level framework for building interactive agentic applications. It is a replacement for AutoGen `v0.2`.

Most of this guide focuses on `v0.4`'s AgentChat API; however, you can also build your own high-level framework using just the Core API.

## New to AutoGen?

Jump straight to the [AgentChat Tutorial](./tutorial/models.ipynb) to get started with `v0.4`.

## What's in this guide?

We provide a detailed guide on how to migrate your existing codebase from `v0.2` to `v0.4`.

See each feature below for detailed information on how to migrate.

- [Migration Guide for v0.2 to v0.4](#migration-guide-for-v02-to-v04)
  - [What is `v0.4`?](#what-is-v04)
  - [New to AutoGen?](#new-to-autogen)
  - [What's in this guide?](#whats-in-this-guide)
  - [Model Client](#model-client)
    - [Use component config](#use-component-config)
    - [Use model client class directly](#use-model-client-class-directly)
  - [Model Client for Ollama-Compatible APIs](#model-client-for-openai-compatible-apis)
  - [Model Client Cache](#model-client-cache)
  - [Assistant Agent](#assistant-agent)
  - [Multi-Modal Agent](#multi-modal-agent)
  - [User Proxy](#user-proxy)
  - [RAG Agent](#rag-agent)
  - [Conversable Agent and Register Reply](#conversable-agent-and-register-reply)
  - [Save and Load Agent State](#save-and-load-agent-state)
  - [Two-Agent Chat](#two-agent-chat)
  - [Tool Use](#tool-use)
  - [Chat Result](#chat-result)
  - [Conversion between v0.2 and v0.4 Messages](#conversion-between-v02-and-v04-messages)
  - [Group Chat](#group-chat)
  - [Group Chat with Resume](#group-chat-with-resume)
  - [Save and Load Group Chat State](#save-and-load-group-chat-state)
  - [Group Chat with Tool Use](#group-chat-with-tool-use)
  - [Group Chat with Custom Selector (Stateflow)](#group-chat-with-custom-selector-stateflow)
  - [Nested Chat](#nested-chat)
  - [Sequential Chat](#sequential-chat)
  - [GPTAssistantAgent](#gptassistantagent)
  - [Long Context Handling](#long-context-handling)
  - [Observability and Control](#observability-and-control)
  - [Code Executors](#code-executors)

The following features currently in `v0.2`
will be provided in the future releases of `v0.4.*` versions:

- Model Client Cost [#4835](https://github.com/microsoft/autogen/issues/4835)
- Teachable Agent
- RAG Agent

We will update this guide when the missing features become available.

## Model Client

In `v0.2` you configure the model client as follows, and create the `OllamaWrapper` object.
"""
logger.info("## What is `v0.4`?")


config_list = [
    {"model": "gpt-4o", "api_key": "sk-xxx"},
    {"model": "llama3.1", "api_key": "sk-xxx"},
]

model_client = OllamaWrapper(config_list=config_list)

"""
> **Note**: In AutoGen 0.2, the Ollama client would try configs in the list until one worked. 0.4 instead expects a specfic model configuration to be chosen.

In `v0.4`, we offer two ways to create a model client.

### Use component config

AutoGen 0.4 has a [generic component configuration system](../core-user-guide/framework/component-config.ipynb). Model clients are a great use case for this. See below for how to create an Ollama chat completion client.
"""
logger.info("### Use component config")


config = {
    "provider": "OllamaChatCompletionClient",
    "config": {
        "model": "gpt-4o",
        "api_key": "sk-xxx" # os.environ["...']
    }
}

model_client = ChatCompletionClient.load_component(config)

"""
### Use model client class directly

Open AI:
"""
logger.info("### Use model client class directly")


model_client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096, api_key="sk-xxx")

"""
Azure Ollama:
"""
logger.info("Azure Ollama:")


model_client = AzureOllamaChatCompletionClient(
    azure_deployment="gpt-4o",
    azure_endpoint="https://<your-endpoint>.openai.azure.com/",
    model="llama3.1", request_timeout=300.0, context_window=4096,
    api_version="2024-09-01-preview",
    api_key="sk-xxx",
)

"""
Read more on {py:class}`~autogen_ext.models.openai.OllamaChatCompletionClient`.

## Model Client for Ollama-Compatible APIs

You can use a the `OllamaChatCompletionClient` to connect to an Ollama-Compatible API,
but you need to specify the `base_url` and `model_info`.
"""
logger.info("## Model Client for Ollama-Compatible APIs")


custom_model_client = OllamaChatCompletionClient(
    model="custom-model-name",
    base_url="https://custom-model.com/reset/of/the/path",
    api_key="placeholder",
    model_info={
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
        "structured_output": True,
    },
)

"""
> **Note**: We don't test all the Ollama-Compatible APIs, and many of them
> works differently from the Ollama API even though they may claim to suppor it.
> Please test them before using them.

Read about [Model Clients](./tutorial/models.ipynb)
in AgentChat Tutorial and more detailed information on [Core API Docs](../core-user-guide/components/model-clients.ipynb)

Support for other hosted models will be added in the future.

## Model Client Cache

In `v0.2`, you can set the cache seed through the `cache_seed` parameter in the LLM config.
The cache is enabled by default.
"""
logger.info("## Model Client Cache")

llm_config = {
    "config_list": [{"model": "gpt-4o", "api_key": "sk-xxx"}],
    "seed": 42,
    "temperature": 0,
    "cache_seed": 42,
}

"""
In `v0.4`, the cache is not enabled by default, to use it you need to use a
{py:class}`~autogen_ext.models.cache.ChatCompletionCache` wrapper around the model client.

You can use a {py:class}`~autogen_ext.cache_store.diskcache.DiskCacheStore` or {py:class}`~autogen_ext.cache_store.redis.RedisStore` to store the cache.
"""
logger.info("In `v0.4`, the cache is not enabled by default, to use it you need to use a")

pip install -U "autogen-ext[openai, diskcache, redis]"

"""
Here's an example of using `diskcache` for local caching:
"""
logger.info("Here's an example of using `diskcache` for local caching:")




async def main():
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


asyncio.run(main())

"""
## Assistant Agent

In `v0.2`, you create an assistant agent as follows:
"""
logger.info("## Assistant Agent")


llm_config = {
    "config_list": [{"model": "gpt-4o", "api_key": "sk-xxx"}],
    "seed": 42,
    "temperature": 0,
}

assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
)

"""
In `v0.4`, it is similar, but you need to specify `model_client` instead of `llm_config`.
"""
logger.info("In `v0.4`, it is similar, but you need to specify `model_client` instead of `llm_config`.")


model_client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096, api_key="sk-xxx", seed=42, temperature=0)

assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    model_client=model_client,
)

"""
However, the usage is somewhat different. In `v0.4`, instead of calling `assistant.send`,
you call `assistant.on_messages` or `assistant.on_messages_stream` to handle incoming messages.
Furthermore, the `on_messages` and `on_messages_stream` methods are asynchronous,
and the latter returns an async generator to stream the inner thoughts of the agent.

Here is how you can call the assistant agent in `v0.4` directly, continuing from the above example:
"""
logger.info("However, the usage is somewhat different. In `v0.4`, instead of calling `assistant.send`,")


async def main() -> None:
    model_client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096, seed=42, temperature=0)

    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        model_client=model_client,
    )

    cancellation_token = CancellationToken()
    async def run_async_code_08058db5():
        async def run_async_code_66be3e7c():
            response = await assistant.on_messages([TextMessage(content="Hello!", source="user")], cancellation_token)
            return response
        response = asyncio.run(run_async_code_66be3e7c())
        logger.success(format_json(response))
        return response
    response = asyncio.run(run_async_code_08058db5())
    logger.success(format_json(response))
    logger.debug(response)

    async def run_async_code_3902376f():
        await model_client.close()
        return 
     = asyncio.run(run_async_code_3902376f())
    logger.success(format_json())

asyncio.run(main())

"""
The {py:class}`~autogen_core.CancellationToken` can be used to cancel the request asynchronously
when you call `cancellation_token.cancel()`, which will cause the `await`
on the `on_messages` call to raise a `CancelledError`.

Read more on [Agent Tutorial](./tutorial/agents.ipynb)
and {py:class}`~autogen_agentchat.agents.AssistantAgent`.

## Multi-Modal Agent

The {py:class}`~autogen_agentchat.agents.AssistantAgent` in `v0.4` supports multi-modal inputs if the model client supports it.
The `vision` capability of the model client is used to determine if the agent supports multi-modal inputs.
"""
logger.info("## Multi-Modal Agent")


async def main() -> None:
    model_client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096, seed=42, temperature=0)

    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        model_client=model_client,
    )

    cancellation_token = CancellationToken()
    message = MultiModalMessage(
        content=["Here is an image:", Image.from_file(Path("test.png"))],
        source="user",
    )
    async def run_async_code_258389fc():
        async def run_async_code_7a6df387():
            response = await assistant.on_messages([message], cancellation_token)
            return response
        response = asyncio.run(run_async_code_7a6df387())
        logger.success(format_json(response))
        return response
    response = asyncio.run(run_async_code_258389fc())
    logger.success(format_json(response))
    logger.debug(response)

    async def run_async_code_3902376f():
        await model_client.close()
        return 
     = asyncio.run(run_async_code_3902376f())
    logger.success(format_json())

asyncio.run(main())

"""
## User Proxy

In `v0.2`, you create a user proxy as follows:
"""
logger.info("## User Proxy")


user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config=False,
    llm_config=False,
)

"""
This user proxy would take input from the user through console, and would terminate
if the incoming message ends with "TERMINATE".

In `v0.4`, a user proxy is simply an agent that takes user input only, there is no
other special configuration needed. You can create a user proxy as follows:
"""
logger.info("This user proxy would take input from the user through console, and would terminate")


user_proxy = UserProxyAgent("user_proxy")

"""
See {py:class}`~autogen_agentchat.agents.UserProxyAgent`
for more details and how to customize the input function with timeout.

## RAG Agent

In `v0.2`, there was the concept of teachable agents as well as a RAG agents that could take a database config.
"""
logger.info("## RAG Agent")

teachable_agent = ConversableAgent(
    name="teachable_agent",
    llm_config=llm_config
)

teachability = Teachability(
    reset_db=False,
    path_to_db_dir="./tmp/interactive/teachability_db"
)

teachability.add_to_agent(teachable_agent)

"""
In `v0.4`, you can implement a RAG agent using the {py:class}`~autogen_core.memory.Memory` class. Specifically, you can define a memory store class, and pass that as a parameter to the assistant agent. See the [Memory](memory.ipynb) tutorial for more details.

This clear separation of concerns allows you to implement a memory store that uses any database or storage system you want (you have to inherit from the `Memory` class) and use it with an assistant agent. The example below shows how to use a ChromaDB vector memory store with the assistant agent. In addition, your application logic should determine how and when to add content to the memory store. For example, you may choose to call `memory.add` for every response from the assistant agent or use a separate LLM call to determine if the content should be added to the memory store.
"""
logger.info("In `v0.4`, you can implement a RAG agent using the {py:class}`~autogen_core.memory.Memory` class. Specifically, you can define a memory store class, and pass that as a parameter to the assistant agent. See the [Memory](memory.ipynb) tutorial for more details.")

chroma_user_memory = ChromaDBVectorMemory(
    config=PersistentChromaDBVectorMemoryConfig(
        collection_name="preferences",
        persistence_path=os.path.join(str(Path.home()), ".chromadb_autogen"),
        k=2,  # Return top  k results
        score_threshold=0.4,  # Minimum similarity score
    )
)


assistant_agent = AssistantAgent(
    name="assistant_agent",
    model_client=OllamaChatCompletionClient(
        model="llama3.1", request_timeout=300.0, context_window=4096,
    ),
    tools=[get_weather],
    memory=[chroma_user_memory],
)

"""
## Conversable Agent and Register Reply

In `v0.2`, you can create a conversable agent and register a reply function as follows:
"""
logger.info("## Conversable Agent and Register Reply")


llm_config = {
    "config_list": [{"model": "gpt-4o", "api_key": "sk-xxx"}],
    "seed": 42,
    "temperature": 0,
}

conversable_agent = ConversableAgent(
    name="conversable_agent",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
    code_execution_config={"work_dir": "coding"},
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
)

def reply_func(
    recipient: ConversableAgent,
    messages: Optional[List[Dict]] = None,
    sender: Optional[Agent] = None,
    config: Optional[Any] = None,
) -> Tuple[bool, Union[str, Dict, None]]:
    return True, "Custom reply"

conversable_agent.register_reply([ConversableAgent], reply_func, position=0)

"""
Rather than guessing what the `reply_func` does, all its parameters,
and what the `position` should be, in `v0.4`, we can simply create a custom agent
and implement the `on_messages`, `on_reset`, and `produced_message_types` methods.
"""
logger.info("Rather than guessing what the `reply_func` does, all its parameters,")


class CustomAgent(BaseChatAgent):
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        return Response(chat_message=TextMessage(content="Custom reply", source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

"""
You can then use the custom agent in the same way as the {py:class}`~autogen_agentchat.agents.AssistantAgent`.
See [Custom Agent Tutorial](custom-agents.ipynb)
for more details.

## Save and Load Agent State

In `v0.2` there is no built-in way to save and load an agent's state: you need
to implement it yourself by exporting the `chat_messages` attribute of `ConversableAgent`
and importing it back through the `chat_messages` parameter.

In `v0.4`, you can call `save_state` and `load_state` methods on agents to save and load their state.
"""
logger.info("## Save and Load Agent State")


async def main() -> None:
    model_client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096, seed=42, temperature=0)

    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        model_client=model_client,
    )

    cancellation_token = CancellationToken()
    async def run_async_code_08058db5():
        async def run_async_code_66be3e7c():
            response = await assistant.on_messages([TextMessage(content="Hello!", source="user")], cancellation_token)
            return response
        response = asyncio.run(run_async_code_66be3e7c())
        logger.success(format_json(response))
        return response
    response = asyncio.run(run_async_code_08058db5())
    logger.success(format_json(response))
    logger.debug(response)

    async def run_async_code_9342c5c4():
        async def run_async_code_06b0c5e6():
            state = await assistant.save_state()
            return state
        state = asyncio.run(run_async_code_06b0c5e6())
        logger.success(format_json(state))
        return state
    state = asyncio.run(run_async_code_9342c5c4())
    logger.success(format_json(state))

    with open("assistant_state.json", "w") as f:
        json.dump(state, f)

    with open("assistant_state.json", "r") as f:
        state = json.load(f)
        logger.debug(state) # Inspect the state, which contains the chat history.

    async def run_async_code_5eae697e():
        async def run_async_code_e9e79ea9():
            response = await assistant.on_messages([TextMessage(content="Tell me a joke.", source="user")], cancellation_token)
            return response
        response = asyncio.run(run_async_code_e9e79ea9())
        logger.success(format_json(response))
        return response
    response = asyncio.run(run_async_code_5eae697e())
    logger.success(format_json(response))
    logger.debug(response)

    async def run_async_code_f181199a():
        await assistant.load_state(state)
        return 
     = asyncio.run(run_async_code_f181199a())
    logger.success(format_json())

    async def run_async_code_5eae697e():
        async def run_async_code_e9e79ea9():
            response = await assistant.on_messages([TextMessage(content="Tell me a joke.", source="user")], cancellation_token)
            return response
        response = asyncio.run(run_async_code_e9e79ea9())
        logger.success(format_json(response))
        return response
    response = asyncio.run(run_async_code_5eae697e())
    logger.success(format_json(response))
    async def run_async_code_3902376f():
        await model_client.close()
        return 
     = asyncio.run(run_async_code_3902376f())
    logger.success(format_json())

asyncio.run(main())

"""
You can also call `save_state` and `load_state` on any teams, such as {py:class}`~autogen_agentchat.teams.RoundRobinGroupChat`
to save and load the state of the entire team.

## Two-Agent Chat

In `v0.2`, you can create a two-agent chat for code execution as follows:
"""
logger.info("## Two-Agent Chat")


llm_config = {
    "config_list": [{"model": "gpt-4o", "api_key": "sk-xxx"}],
    "seed": 42,
    "temperature": 0,
}

assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant. Write all code in python. Reply only 'TERMINATE' if the task is done.",
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"code_executor": LocalCommandLineCodeExecutor(work_dir="coding")},
    llm_config=False,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

chat_result = user_proxy.initiate_chat(assistant, message="Write a python script to print 'Hello, world!'")
logger.debug(chat_result)

"""
To get the same behavior in `v0.4`, you can use the {py:class}`~autogen_agentchat.agents.AssistantAgent`
and {py:class}`~autogen_agentchat.agents.CodeExecutorAgent` together in a {py:class}`~autogen_agentchat.teams.RoundRobinGroupChat`.
"""
logger.info("To get the same behavior in `v0.4`, you can use the {py:class}`~autogen_agentchat.agents.AssistantAgent`")


async def main() -> None:
    model_client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096, seed=42, temperature=0)

    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant. Write all code in python. Reply only 'TERMINATE' if the task is done.",
        model_client=model_client,
    )

    code_executor = CodeExecutorAgent(
        name="code_executor",
        code_executor=LocalCommandLineCodeExecutor(work_dir="coding"),
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)

    group_chat = RoundRobinGroupChat([assistant, code_executor], termination_condition=termination)

    stream = group_chat.run_stream(task="Write a python script to print 'Hello, world!'")
    async def run_async_code_8cdf6b5b():
        await Console(stream)
        return 
     = asyncio.run(run_async_code_8cdf6b5b())
    logger.success(format_json())

    async def run_async_code_3902376f():
        await model_client.close()
        return 
     = asyncio.run(run_async_code_3902376f())
    logger.success(format_json())

asyncio.run(main())

"""
## Tool Use

In `v0.2`, to create a tool use chatbot, you must have two agents, one for calling the tool and one for executing the tool.
You need to initiate a two-agent chat for every user request.
"""
logger.info("## Tool Use")


llm_config = {
    "config_list": [{"model": "gpt-4o", "api_key": "sk-xxx"}],
    "seed": 42,
    "temperature": 0,
}

tool_caller = AssistantAgent(
    name="tool_caller",
    system_message="You are a helpful assistant. You can call tools to help user.",
    llm_config=llm_config,
    max_consecutive_auto_reply=1, # Set to 1 so that we return to the application after each assistant reply as we are building a chatbot.
)

tool_executor = UserProxyAgent(
    name="tool_executor",
    human_input_mode="NEVER",
    code_execution_config=False,
    llm_config=False,
)

def get_weather(city: str) -> str:
    return f"The weather in {city} is 72 degree and sunny."

register_function(get_weather, caller=tool_caller, executor=tool_executor)

while True:
    user_input = input("User: ")
    if user_input == "exit":
        break
    chat_result = tool_executor.initiate_chat(
        tool_caller,
        message=user_input,
        summary_method="reflection_with_llm", # To let the model reflect on the tool use, set to "last_msg" to return the tool call result directly.
    )
    logger.debug("Assistant:", chat_result.summary)

"""
In `v0.4`, you really just need one agent -- the {py:class}`~autogen_agentchat.agents.AssistantAgent` -- to handle
both the tool calling and tool execution.
"""
logger.info("In `v0.4`, you really just need one agent -- the {py:class}`~autogen_agentchat.agents.AssistantAgent` -- to handle")


def get_weather(city: str) -> str: # Async tool is possible too.
    return f"The weather in {city} is 72 degree and sunny."

async def main() -> None:
    model_client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096, seed=42, temperature=0)
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant. You can call tools to help user.",
        model_client=model_client,
        tools=[get_weather],
        reflect_on_tool_use=True, # Set to True to have the model reflect on the tool use, set to False to return the tool call result directly.
    )
    while True:
        user_input = input("User: ")
        if user_input == "exit":
            break
        async def run_async_code_21ee0b1a():
            async def run_async_code_a821ed14():
                response = await assistant.on_messages([TextMessage(content=user_input, source="user")], CancellationToken())
                return response
            response = asyncio.run(run_async_code_a821ed14())
            logger.success(format_json(response))
            return response
        response = asyncio.run(run_async_code_21ee0b1a())
        logger.success(format_json(response))
        logger.debug("Assistant:", response.chat_message.to_text())
    async def run_async_code_3902376f():
        await model_client.close()
        return 
     = asyncio.run(run_async_code_3902376f())
    logger.success(format_json())

asyncio.run(main())

"""
When using tool-equipped agents inside a group chat such as
{py:class}`~autogen_agentchat.teams.RoundRobinGroupChat`,
you simply do the same as above to add tools to the agents, and create a
group chat with the agents.

## Chat Result

In `v0.2`, you get a `ChatResult` object from the `initiate_chat` method.
For example:
"""
logger.info("## Chat Result")

chat_result = tool_executor.initiate_chat(
    tool_caller,
    message=user_input,
    summary_method="reflection_with_llm",
)
logger.debug(chat_result.summary) # Get LLM-reflected summary of the chat.
logger.debug(chat_result.chat_history) # Get the chat history.
logger.debug(chat_result.cost) # Get the cost of the chat.
logger.debug(chat_result.human_input) # Get the human input solicited by the chat.

"""
See [ChatResult Docs](https://microsoft.github.io/autogen/0.2/docs/reference/agentchat/chat#chatresult)
for more details.

In `v0.4`, you get a {py:class}`~autogen_agentchat.base.TaskResult` object from a `run` or `run_stream` method.
The {py:class}`~autogen_agentchat.base.TaskResult` object contains the `messages` which is the message history
of the chat, including both agents' private (tool calls, etc.) and public messages.

There are some notable differences between {py:class}`~autogen_agentchat.base.TaskResult` and `ChatResult`:

- The `messages` list in {py:class}`~autogen_agentchat.base.TaskResult` uses different message format than the `ChatResult.chat_history` list.
- There is no `summary` field. It is up to the application to decide how to summarize the chat using the `messages` list.
- `human_input` is not provided in the {py:class}`~autogen_agentchat.base.TaskResult` object, as the user input can be extracted from the `messages` list by filtering with the `source` field.
- `cost` is not provided in the {py:class}`~autogen_agentchat.base.TaskResult` object, however, you can calculate the cost based on token usage. It would be a great community extension to add cost calculation. See [community extensions](../extensions-user-guide/discover.md).

## Conversion between v0.2 and v0.4 Messages

You can use the following conversion functions to convert between a v0.4 message in
{py:attr}`autogen_agentchat.base.TaskResult.messages` and a v0.2 message in `ChatResult.chat_history`.
"""
logger.info("## Conversion between v0.2 and v0.4 Messages")




def convert_to_v02_message(
    message: BaseAgentEvent | BaseChatMessage,
    role: Literal["assistant", "user", "tool"],
    image_detail: Literal["auto", "high", "low"] = "auto",
) -> Dict[str, Any]:
    """Convert a v0.4 AgentChat message to a v0.2 message.

    Args:
        message (BaseAgentEvent | BaseChatMessage): The message to convert.
        role (Literal["assistant", "user", "tool"]): The role of the message.
        image_detail (Literal["auto", "high", "low"], optional): The detail level of image content in multi-modal message. Defaults to "auto".

    Returns:
        Dict[str, Any]: The converted AutoGen v0.2 message.
    """
    v02_message: Dict[str, Any] = {}
    if isinstance(message, TextMessage | StopMessage | HandoffMessage | ToolCallSummaryMessage):
        v02_message = {"content": message.content, "role": role, "name": message.source}
    elif isinstance(message, MultiModalMessage):
        v02_message = {"content": [], "role": role, "name": message.source}
        for modal in message.content:
            if isinstance(modal, str):
                v02_message["content"].append({"type": "text", "text": modal})
            elif isinstance(modal, Image):
                v02_message["content"].append(modal.to_openai_format(detail=image_detail))
            else:
                raise ValueError(f"Invalid multimodal message content: {modal}")
    elif isinstance(message, ToolCallRequestEvent):
        v02_message = {"tool_calls": [], "role": "assistant", "content": None, "name": message.source}
        for tool_call in message.content:
            v02_message["tool_calls"].append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {"name": tool_call.name, "args": tool_call.arguments},
                }
            )
    elif isinstance(message, ToolCallExecutionEvent):
        tool_responses: List[Dict[str, str]] = []
        for tool_result in message.content:
            tool_responses.append(
                {
                    "tool_call_id": tool_result.call_id,
                    "role": "tool",
                    "content": tool_result.content,
                }
            )
        content = "\n\n".join([response["content"] for response in tool_responses])
        v02_message = {"tool_responses": tool_responses, "role": "tool", "content": content}
    else:
        raise ValueError(f"Invalid message type: {type(message)}")
    return v02_message


def convert_to_v04_message(message: Dict[str, Any]) -> BaseAgentEvent | BaseChatMessage:
    """Convert a v0.2 message to a v0.4 AgentChat message."""
    if "tool_calls" in message:
        tool_calls: List[FunctionCall] = []
        for tool_call in message["tool_calls"]:
            tool_calls.append(
                FunctionCall(
                    id=tool_call["id"],
                    name=tool_call["function"]["name"],
                    arguments=tool_call["function"]["args"],
                )
            )
        return ToolCallRequestEvent(source=message["name"], content=tool_calls)
    elif "tool_responses" in message:
        tool_results: List[FunctionExecutionResult] = []
        for tool_response in message["tool_responses"]:
            tool_results.append(
                FunctionExecutionResult(
                    call_id=tool_response["tool_call_id"],
                    content=tool_response["content"],
                    is_error=False,
                    name=tool_response["name"],
                )
            )
        return ToolCallExecutionEvent(source="tools", content=tool_results)
    elif isinstance(message["content"], list):
        content: List[str | Image] = []
        for modal in message["content"]:  # type: ignore
            if modal["type"] == "text":  # type: ignore
                content.append(modal["text"])  # type: ignore
            else:
                content.append(Image.from_uri(modal["image_url"]["url"]))  # type: ignore
        return MultiModalMessage(content=content, source=message["name"])
    elif isinstance(message["content"], str):
        return TextMessage(content=message["content"], source=message["name"])
    else:
        raise ValueError(f"Unable to convert message: {message}")

"""
## Group Chat

In `v0.2`, you need to create a `GroupChat` class and pass it into a
`GroupChatManager`, and have a participant that is a user proxy to initiate the chat.
For a simple scenario of a writer and a critic, you can do the following:
"""
logger.info("## Group Chat")


llm_config = {
    "config_list": [{"model": "gpt-4o", "api_key": "sk-xxx"}],
    "seed": 42,
    "temperature": 0,
}

writer = AssistantAgent(
    name="writer",
    description="A writer.",
    system_message="You are a writer.",
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("APPROVE"),
)

critic = AssistantAgent(
    name="critic",
    description="A critic.",
    system_message="You are a critic, provide feedback on the writing. Reply only 'APPROVE' if the task is done.",
    llm_config=llm_config,
)

groupchat = GroupChat(agents=[writer, critic], messages=[], max_round=12)

manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config, speaker_selection_method="round_robin")

result = editor.initiate_chat(
    manager,
    message="Write a short story about a robot that discovers it has feelings.",
)
logger.debug(result.summary)

"""
In `v0.4`, you can use the {py:class}`~autogen_agentchat.teams.RoundRobinGroupChat` to achieve the same behavior.
"""
logger.info("In `v0.4`, you can use the {py:class}`~autogen_agentchat.teams.RoundRobinGroupChat` to achieve the same behavior.")


async def main() -> None:
    model_client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096, seed=42, temperature=0)

    writer = AssistantAgent(
        name="writer",
        description="A writer.",
        system_message="You are a writer.",
        model_client=model_client,
    )

    critic = AssistantAgent(
        name="critic",
        description="A critic.",
        system_message="You are a critic, provide feedback on the writing. Reply only 'APPROVE' if the task is done.",
        model_client=model_client,
    )

    termination = TextMentionTermination("APPROVE")

    group_chat = RoundRobinGroupChat([writer, critic], termination_condition=termination, max_turns=12)

    stream = group_chat.run_stream(task="Write a short story about a robot that discovers it has feelings.")
    async def run_async_code_8cdf6b5b():
        await Console(stream)
        return 
     = asyncio.run(run_async_code_8cdf6b5b())
    logger.success(format_json())
    async def run_async_code_3902376f():
        await model_client.close()
        return 
     = asyncio.run(run_async_code_3902376f())
    logger.success(format_json())

asyncio.run(main())

"""
For LLM-based speaker selection, you can use the {py:class}`~autogen_agentchat.teams.SelectorGroupChat` instead.
See [Selector Group Chat Tutorial](./selector-group-chat.ipynb)
and {py:class}`~autogen_agentchat.teams.SelectorGroupChat` for more details.

> **Note**: In `v0.4`, you do not need to register functions on a user proxy to use tools
> in a group chat. You can simply pass the tool functions to the {py:class}`~autogen_agentchat.agents.AssistantAgent` as shown in the [Tool Use](#tool-use) section.
> The agent will automatically call the tools when needed.
> If your tool doesn't output well formed response, you can use the `reflect_on_tool_use` parameter to have the model reflect on the tool use.

## Group Chat with Resume

In `v0.2`, group chat with resume is a bit complicated. You need to explicitly
save the group chat messages and load them back when you want to resume the chat.
See [Resuming Group Chat in v0.2](https://microsoft.github.io/autogen/0.2/docs/topics/groupchat/resuming_groupchat) for more details.

In `v0.4`, you can simply call `run` or `run_stream` again with the same group chat object to resume the chat. To export and load the state, you can use
`save_state` and `load_state` methods.
"""
logger.info("## Group Chat with Resume")


def create_team(model_client : OllamaChatCompletionClient) -> RoundRobinGroupChat:
    writer = AssistantAgent(
        name="writer",
        description="A writer.",
        system_message="You are a writer.",
        model_client=model_client,
    )

    critic = AssistantAgent(
        name="critic",
        description="A critic.",
        system_message="You are a critic, provide feedback on the writing. Reply only 'APPROVE' if the task is done.",
        model_client=model_client,
    )

    termination = TextMentionTermination("APPROVE")

    group_chat = RoundRobinGroupChat([writer, critic], termination_condition=termination)

    return group_chat


async def main() -> None:
    model_client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096, seed=42, temperature=0)
    group_chat = create_team(model_client)

    stream = group_chat.run_stream(task="Write a short story about a robot that discovers it has feelings.")
    async def run_async_code_8cdf6b5b():
        await Console(stream)
        return 
     = asyncio.run(run_async_code_8cdf6b5b())
    logger.success(format_json())

    async def run_async_code_b57dcd34():
        async def run_async_code_ca3cc5e1():
            state = await group_chat.save_state()
            return state
        state = asyncio.run(run_async_code_ca3cc5e1())
        logger.success(format_json(state))
        return state
    state = asyncio.run(run_async_code_b57dcd34())
    logger.success(format_json(state))
    with open("group_chat_state.json", "w") as f:
        json.dump(state, f)

    group_chat = create_team(model_client)

    with open("group_chat_state.json", "r") as f:
        state = json.load(f)
    async def run_async_code_116107b1():
        await group_chat.load_state(state)
        return 
     = asyncio.run(run_async_code_116107b1())
    logger.success(format_json())

    stream = group_chat.run_stream(task="Translate the story into Chinese.")
    async def run_async_code_8cdf6b5b():
        await Console(stream)
        return 
     = asyncio.run(run_async_code_8cdf6b5b())
    logger.success(format_json())

    async def run_async_code_3902376f():
        await model_client.close()
        return 
     = asyncio.run(run_async_code_3902376f())
    logger.success(format_json())

asyncio.run(main())

"""
## Save and Load Group Chat State

In `v0.2`, you need to explicitly save the group chat messages and load them back when you want to resume the chat.

In `v0.4`, you can simply call `save_state` and `load_state` methods on the group chat object.
See [Group Chat with Resume](#group-chat-with-resume) for an example.

## Group Chat with Tool Use

In `v0.2` group chat, when tools are involved, you need to register the tool functions on a user proxy,
and include the user proxy in the group chat. The tool calls made by other agents
will be routed to the user proxy to execute.

We have observed numerous issues with this approach, such as the the tool call
routing not working as expected, and the tool call request and result cannot be
accepted by models without support for function calling.

In `v0.4`, there is no need to register the tool functions on a user proxy,
as the tools are directly executed within the {py:class}`~autogen_agentchat.agents.AssistantAgent`,
which publishes the response from the tool to the group chat.
So the group chat manager does not need to be involved in routing tool calls.

See [Selector Group Chat Tutorial](./selector-group-chat.ipynb) for an example
of using tools in a group chat.

## Group Chat with Custom Selector (Stateflow)

In `v0.2` group chat, when the `speaker_selection_method` is set to a custom function,
it can override the default selection method. This is useful for implementing
a state-based selection method.
For more details, see [Custom Sepaker Selection in v0.2](https://microsoft.github.io/autogen/0.2/docs/topics/groupchat/customized_speaker_selection).

In `v0.4`, you can use the {py:class}`~autogen_agentchat.teams.SelectorGroupChat` with `selector_func` to achieve the same behavior.
The `selector_func` is a function that takes the current message thread of the group chat
and returns the next speaker's name. If `None` is returned, the LLM-based
selection method will be used.

Here is an example of using the state-based selection method to implement
a web search/analysis scenario.
"""
logger.info("## Save and Load Group Chat State")


def search_web_tool(query: str) -> str:
    if "2006-2007" in query:
        return """Here are the total points scored by Miami Heat players in the 2006-2007 season:
        Udonis Haslem: 844 points
        Dwayne Wade: 1397 points
        James Posey: 550 points
        ...
        """
    elif "2007-2008" in query:
        return "The number of total rebounds for Dwayne Wade in the Miami Heat season 2007-2008 is 214."
    elif "2008-2009" in query:
        return "The number of total rebounds for Dwayne Wade in the Miami Heat season 2008-2009 is 398."
    return "No data found."


def percentage_change_tool(start: float, end: float) -> float:
    return ((end - start) / start) * 100

def create_team(model_client : OllamaChatCompletionClient) -> SelectorGroupChat:
    planning_agent = AssistantAgent(
        "PlanningAgent",
        description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
        model_client=model_client,
        system_message="""
        You are a planning agent.
        Your job is to break down complex tasks into smaller, manageable subtasks.
        Your team members are:
            Web search agent: Searches for information
            Data analyst: Performs calculations

        You only plan and delegate tasks - you do not execute them yourself.

        When assigning tasks, use this format:
        1. <agent> : <task>

        After all tasks are complete, summarize the findings and end with "TERMINATE".
        """,
    )

    web_search_agent = AssistantAgent(
        "WebSearchAgent",
        description="A web search agent.",
        tools=[search_web_tool],
        model_client=model_client,
        system_message="""
        You are a web search agent.
        Your only tool is search_tool - use it to find information.
        You make only one search call at a time.
        Once you have the results, you never do calculations based on them.
        """,
    )

    data_analyst_agent = AssistantAgent(
        "DataAnalystAgent",
        description="A data analyst agent. Useful for performing calculations.",
        model_client=model_client,
        tools=[percentage_change_tool],
        system_message="""
        You are a data analyst.
        Given the tasks you have been assigned, you should analyze the data and provide results using the tools provided.
        """,
    )

    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=25)
    termination = text_mention_termination | max_messages_termination

    def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
        if messages[-1].source != planning_agent.name:
            return planning_agent.name # Always return to the planning agent after the other agents have spoken.
        return None

    team = SelectorGroupChat(
        [planning_agent, web_search_agent, data_analyst_agent],
        model_client=OllamaChatCompletionClient(model="llama3.1"), # Use a smaller model for the selector.
        termination_condition=termination,
        selector_func=selector_func,
    )
    return team

async def main() -> None:
    model_client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096)
    team = create_team(model_client)
    task = "Who was the Miami Heat player with the highest points in the 2006-2007 season, and what was the percentage change in his total rebounds between the 2007-2008 and 2008-2009 seasons?"
    async def run_async_code_289d0f72():
        await Console(team.run_stream(task=task))
        return 
     = asyncio.run(run_async_code_289d0f72())
    logger.success(format_json())

asyncio.run(main())

"""
## Nested Chat

Nested chat allows you to nest a whole team or another agent inside
an agent. This is useful for creating a hierarchical structure of agents
or "information silos", as the nested agents cannot communicate directly
with other agents outside of the same group.

In `v0.2`, nested chat is supported by using the `register_nested_chats` method
on the `ConversableAgent` class.
You need to specify the nested sequence of agents using dictionaries,
See [Nested Chat in v0.2](https://microsoft.github.io/autogen/0.2/docs/tutorial/conversation-patterns#nested-chats)
for more details.

In `v0.4`, nested chat is an implementation detail of a custom agent.
You can create a custom agent that takes a team or another agent as a parameter
and implements the `on_messages` method to trigger the nested team or agent.
It is up to the application to decide how to pass or transform the messages from
and to the nested team or agent.

The following example shows a simple nested chat that counts numbers.
"""
logger.info("## Nested Chat")


class CountingAgent(BaseChatAgent):
    """An agent that returns a new number by adding 1 to the last number in the input messages."""
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        if len(messages) == 0:
            last_number = 0 # Start from 0 if no messages are given.
        else:
            assert isinstance(messages[-1], TextMessage)
            last_number = int(messages[-1].content) # Otherwise, start from the last number.
        return Response(chat_message=TextMessage(content=str(last_number + 1), source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

class NestedCountingAgent(BaseChatAgent):
    """An agent that increments the last number in the input messages
    multiple times using a nested counting team."""
    def __init__(self, name: str, counting_team: RoundRobinGroupChat) -> None:
        super().__init__(name, description="An agent that counts numbers.")
        self._counting_team = counting_team

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        async def run_async_code_31c1c27d():
            async def run_async_code_8b469f05():
                result = await self._counting_team.run(task=messages, cancellation_token=cancellation_token)
                return result
            result = asyncio.run(run_async_code_8b469f05())
            logger.success(format_json(result))
            return result
        result = asyncio.run(run_async_code_31c1c27d())
        logger.success(format_json(result))
        assert isinstance(result.messages[-1], TextMessage)
        return Response(chat_message=result.messages[-1], inner_messages=result.messages[len(messages):-1])

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        async def run_async_code_0d20535f():
            await self._counting_team.reset()
            return 
         = asyncio.run(run_async_code_0d20535f())
        logger.success(format_json())

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

async def main() -> None:
    counting_agent_1 = CountingAgent("counting_agent_1", description="An agent that counts numbers.")
    counting_agent_2 = CountingAgent("counting_agent_2", description="An agent that counts numbers.")
    counting_team = RoundRobinGroupChat([counting_agent_1, counting_agent_2], max_turns=5)
    nested_counting_agent = NestedCountingAgent("nested_counting_agent", counting_team)
    async def run_async_code_4ab53484():
        async def run_async_code_f517fe10():
            response = await nested_counting_agent.on_messages([TextMessage(content="1", source="user")], CancellationToken())
            return response
        response = asyncio.run(run_async_code_f517fe10())
        logger.success(format_json(response))
        return response
    response = asyncio.run(run_async_code_4ab53484())
    logger.success(format_json(response))
    assert response.inner_messages is not None
    for message in response.inner_messages:
        logger.debug(message)
    logger.debug(response.chat_message)

asyncio.run(main())

"""
You should see the following output:
"""
logger.info("You should see the following output:")

source='counting_agent_1' models_usage=None content='2' type='TextMessage'
source='counting_agent_2' models_usage=None content='3' type='TextMessage'
source='counting_agent_1' models_usage=None content='4' type='TextMessage'
source='counting_agent_2' models_usage=None content='5' type='TextMessage'
source='counting_agent_1' models_usage=None content='6' type='TextMessage'

"""
You can take a look at {py:class}`~autogen_agentchat.agents.SocietyOfMindAgent`
for a more complex implementation.

## Sequential Chat

In `v0.2`, sequential chat is supported by using the `initiate_chats` function.
It takes input a list of dictionary configurations for each step of the sequence.
See [Sequential Chat in v0.2](https://microsoft.github.io/autogen/0.2/docs/tutorial/conversation-patterns#sequential-chats)
for more details.

Base on the feedback from the community, the `initiate_chats` function
is too opinionated and not flexible enough to support the diverse set of scenarios that
users want to implement. We often find users struggling to get the `initiate_chats` function
to work when they can easily glue the steps together usign basic Python code.
Therefore, in `v0.4`, we do not provide a built-in function for sequential chat in the AgentChat API.

Instead, you can create an event-driven sequential workflow using the Core API,
and use the other components provided the AgentChat API to implement each step of the workflow.
See an example of sequential workflow in the [Core API Tutorial](../core-user-guide/design-patterns/sequential-workflow.ipynb).

We recognize that the concept of workflow is at the heart of many applications,
and we will provide more built-in support for workflows in the future.

## GPTAssistantAgent

In `v0.2`, `GPTAssistantAgent` is a special agent class that is backed by the Ollama Assistant API.

In `v0.4`, the equivalent is the {py:class}`~autogen_ext.agents.openai.OllamaAssistantAgent` class.
It supports the same set of features as the `GPTAssistantAgent` in `v0.2` with
more such as customizable threads and file uploads.
See {py:class}`~autogen_ext.agents.openai.OllamaAssistantAgent` for more details.

## Long Context Handling

In `v0.2`, long context that overflows the model's context window can be handled
by using the `transforms` capability that is added to an `ConversableAgent`
after which is contructed.

The feedbacks from our community has led us to believe this feature is essential
and should be a built-in component of {py:class}`~autogen_agentchat.agents.AssistantAgent`, and can be used for
every custom agent.

In `v0.4`, we introduce the {py:class}`~autogen_core.model_context.ChatCompletionContext` base class that manages
message history and provides a virtual view of the history. Applications can use
built-in implementations such as {py:class}`~autogen_core.model_context.BufferedChatCompletionContext` to
limit the message history sent to the model, or provide their own implementations
that creates different virtual views.

To use {py:class}`~autogen_core.model_context.BufferedChatCompletionContext` in an {py:class}`~autogen_agentchat.agents.AssistantAgent` in a chatbot scenario.
"""
logger.info("## Sequential Chat")


async def main() -> None:
    model_client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096, seed=42, temperature=0)

    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        model_client=model_client,
        model_context=BufferedChatCompletionContext(buffer_size=10), # Model can only view the last 10 messages.
    )
    while True:
        user_input = input("User: ")
        if user_input == "exit":
            break
        async def run_async_code_21ee0b1a():
            async def run_async_code_a821ed14():
                response = await assistant.on_messages([TextMessage(content=user_input, source="user")], CancellationToken())
                return response
            response = asyncio.run(run_async_code_a821ed14())
            logger.success(format_json(response))
            return response
        response = asyncio.run(run_async_code_21ee0b1a())
        logger.success(format_json(response))
        logger.debug("Assistant:", response.chat_message.to_text())

    async def run_async_code_3902376f():
        await model_client.close()
        return 
     = asyncio.run(run_async_code_3902376f())
    logger.success(format_json())

asyncio.run(main())

"""
In this example, the chatbot can only read the last 10 messages in the history.

## Observability and Control

In `v0.4` AgentChat, you can observe the agents by using the `on_messages_stream` method
which returns an async generator to stream the inner thoughts and actions of the agent.
For teams, you can use the `run_stream` method to stream the inner conversation among the agents in the team.
Your application can use these streams to observe the agents and teams in real-time.

Both the `on_messages_stream` and `run_stream` methods takes a {py:class}`~autogen_core.CancellationToken` as a parameter
which can be used to cancel the output stream asynchronously and stop the agent or team.
For teams, you can also use termination conditions to stop the team when a certain condition is met.
See [Termination Condition Tutorial](./tutorial/termination.ipynb)
for more details.

Unlike the `v0.2` which comes with a special logging module, the `v0.4` API
simply uses Python's `logging` module to log events such as model client calls.
See [Logging](../core-user-guide/framework/logging.md)
in the Core API documentation for more details.

## Code Executors

The code executors in `v0.2` and `v0.4` are nearly identical except
the `v0.4` executors support async API. You can also use
{py:class}`~autogen_core.CancellationToken` to cancel a code execution if it takes too long.
See [Command Line Code Executors Tutorial](../core-user-guide/components/command-line-code-executors.ipynb)
in the Core API documentation.

We also added `ACADynamicSessionsCodeExecutor` that can use Azure Container Apps (ACA)
dynamic sessions for code execution.
See [ACA Dynamic Sessions Code Executor Docs](../extensions-user-guide/azure-container-code-executor.ipynb).
"""
logger.info("## Observability and Control")

logger.info("\n\n[DONE]", bright=True)