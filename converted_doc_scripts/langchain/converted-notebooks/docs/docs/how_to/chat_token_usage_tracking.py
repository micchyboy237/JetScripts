from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# How to track token usage in ChatModels

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Chat models](/docs/concepts/chat_models)

:::

Tracking [token](/docs/concepts/tokens/) usage to calculate cost is an important part of putting your app in production. This guide goes over how to obtain this information from your LangChain model calls.

This guide requires `langchain-anthropic` and `langchain-ollama >= 0.3.11`.
"""
logger.info("# How to track token usage in ChatModels")

# %pip install -qU langchain-anthropic langchain-ollama

"""
:::note A note on streaming with Ollama

Ollama's Chat Completions API does not stream token usage statistics by default (see API reference
[here](https://platform.ollama.com/docs/api-reference/completions/create#completions-create-stream_options)).
To recover token counts when streaming with `ChatOllama` or `AzureChatOllama`, set `stream_usage=True` as
demonstrated in this guide.

:::

## Using LangSmith

You can use [LangSmith](https://www.langchain.com/langsmith) to help track token usage in your LLM application. See the [LangSmith quick start guide](https://docs.smith.langchain.com/).

## Using AIMessage.usage_metadata

A number of model providers return token usage information as part of the chat generation response. When available, this information will be included on the `AIMessage` objects produced by the corresponding model.

LangChain `AIMessage` objects include a [usage_metadata](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessage.html#langchain_core.messages.ai.AIMessage.usage_metadata) attribute. When populated, this attribute will be a [UsageMetadata](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.UsageMetadata.html) dictionary with standard keys (e.g., `"input_tokens"` and `"output_tokens"`). They will also include information on cached token usage and tokens from multi-modal data.

Examples:

**Ollama**:
"""
logger.info("## Using LangSmith")


llm = init_chat_model(model="llama3.2")
ollama_response = llm.invoke("hello")
ollama_response.usage_metadata

"""
**Ollama**:
"""


llm = ChatOllama(model="llama3.2")
anthropic_response = llm.invoke("hello")
anthropic_response.usage_metadata

"""
### Streaming

Some providers support token count metadata in a streaming context.

#### Ollama

For example, Ollama will return a message [chunk](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessageChunk.html) at the end of a stream with token usage information. This behavior is supported by `langchain-ollama >= 0.1.9` and can be enabled by setting `stream_usage=True`. This attribute can also be set when `ChatOllama` is instantiated.

:::note
By default, the last message chunk in a stream will include a `"finish_reason"` in the message's `response_metadata` attribute. If we include token usage in streaming mode, an additional chunk containing usage metadata will be added to the end of the stream, such that `"finish_reason"` appears on the second to last message chunk.
:::
"""
logger.info("### Streaming")

llm = init_chat_model(model="llama3.2")

aggregate = None
for chunk in llm.stream("hello", stream_usage=True):
    logger.debug(chunk)
    aggregate = chunk if aggregate is None else aggregate + chunk

"""
Note that the usage metadata will be included in the sum of the individual message chunks:
"""
logger.info("Note that the usage metadata will be included in the sum of the individual message chunks:")

logger.debug(aggregate.content)
logger.debug(aggregate.usage_metadata)

"""
To disable streaming token counts for Ollama, set `stream_usage` to False, or omit it from the parameters:
"""
logger.info("To disable streaming token counts for Ollama, set `stream_usage` to False, or omit it from the parameters:")

aggregate = None
for chunk in llm.stream("hello"):
    logger.debug(chunk)

"""
You can also enable streaming token usage by setting `stream_usage` when instantiating the chat model. This can be useful when incorporating chat models into LangChain [chains](/docs/concepts/lcel): usage metadata can be monitored when [streaming intermediate steps](/docs/how_to/streaming#using-stream-events) or using tracing software such as [LangSmith](https://docs.smith.langchain.com/).

See the below example, where we return output structured to a desired schema, but can still observe token usage streamed from intermediate steps.
"""
logger.info("You can also enable streaming token usage by setting `stream_usage` when instantiating the chat model. This can be useful when incorporating chat models into LangChain [chains](/docs/concepts/lcel): usage metadata can be monitored when [streaming intermediate steps](/docs/how_to/streaming#using-stream-events) or using tracing software such as [LangSmith](https://docs.smith.langchain.com/).")



class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


llm = init_chat_model(
    model="llama3.2",
    stream_usage=True,
)
structured_llm = llm.with_structured_output(Joke)

for event in structured_llm.stream_events("Tell me a joke"):
    if event["event"] == "on_chat_model_end":
        logger.debug(f"Token usage: {event['data']['output'].usage_metadata}\n")
    elif event["event"] == "on_chain_end" and event["name"] == "RunnableSequence":
        logger.debug(event["data"]["output"])
    else:
        pass

"""
Token usage is also visible in the corresponding [LangSmith trace](https://smith.langchain.com/public/fe6513d5-7212-4045-82e0-fefa28bc7656/r) in the payload from the chat model.

## Using callbacks

:::info Requires ``langchain-core>=0.3.49``

:::

LangChain implements a callback handler and context manager that will track token usage across calls of any chat model that returns `usage_metadata`.

There are also some API-specific callback context managers that maintain pricing for different models, allowing for cost estimation in real time. They are currently only implemented for the Ollama API and Bedrock Ollama API, and are available in `langchain-community`:

- [get_ollama_callback](https://python.langchain.com/api_reference/community/callbacks/langchain_community.callbacks.manager.get_ollama_callback.html)
- [get_bedrock_anthropic_callback](https://python.langchain.com/api_reference/community/callbacks/langchain_community.callbacks.manager.get_bedrock_anthropic_callback.html)

Below, we demonstrate the general-purpose usage metadata callback manager. We can track token usage through configuration or as a context manager.

### Tracking token usage through configuration

To track token usage through configuration, instantiate a `UsageMetadataCallbackHandler` and pass it into the config:
"""
logger.info("## Using callbacks")


llm_1 = init_chat_model(model="ollama:llama3.2")
llm_2 = init_chat_model(model="ollama:claude-3-5-haiku-latest")

callback = UsageMetadataCallbackHandler()
result_1 = llm_1.invoke("Hello", config={"callbacks": [callback]})
result_2 = llm_2.invoke("Hello", config={"callbacks": [callback]})
callback.usage_metadata

"""
### Tracking token usage using a context manager

You can also use `get_usage_metadata_callback` to create a context manager and aggregate usage metadata there:
"""
logger.info("### Tracking token usage using a context manager")


llm_1 = init_chat_model(model="ollama:llama3.2")
llm_2 = init_chat_model(model="ollama:claude-3-5-haiku-latest")

with get_usage_metadata_callback() as cb:
    llm_1.invoke("Hello")
    llm_2.invoke("Hello")
    logger.debug(cb.usage_metadata)

"""
Either of these methods will aggregate token usage across multiple calls to each model. For example, you can use it in an [agent](https://python.langchain.com/docs/concepts/agents/) to track token usage across repeated calls to one model:
"""
logger.info("Either of these methods will aggregate token usage across multiple calls to each model. For example, you can use it in an [agent](https://python.langchain.com/docs/concepts/agents/) to track token usage across repeated calls to one model:")

# %pip install -qU langgraph



def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return "It's sunny."


callback = UsageMetadataCallbackHandler()

tools = [get_weather]
agent = create_react_agent("ollama:llama3.2", tools)
for step in agent.stream(
    {"messages": [{"role": "user", "content": "What's the weather in Boston?"}]},
    stream_mode="values",
    config={"callbacks": [callback]},
):
    step["messages"][-1].pretty_logger.debug()


logger.debug(f"\nTotal usage: {callback.usage_metadata}")

"""
## Next steps

You've now seen a few examples of how to track token usage for supported providers.

Next, check out the other how-to guides chat models in this section, like [how to get a model to return structured output](/docs/how_to/structured_output) or [how to add caching to your chat models](/docs/how_to/chat_model_caching).
"""
logger.info("## Next steps")


logger.info("\n\n[DONE]", bright=True)