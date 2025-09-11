from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
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
# How to parse text from message objects

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Chat models](/docs/concepts/chat_models/)
- [Messages](/docs/concepts/messages/)
- [Output parsers](/docs/concepts/output_parsers/)
- [LangChain Expression Language (LCEL)](/docs/concepts/lcel/)

:::

LangChain [message](/docs/concepts/messages/) objects support content in a [variety of formats](/docs/concepts/messages/#content), including text, [multimodal data](/docs/concepts/multimodality/), and a list of [content block](/docs/concepts/messages/#aimessage) dicts.

The format of [Chat model](/docs/concepts/chat_models/) response content may depend on the provider. For example, the chat model for [Ollama](/docs/integrations/chat/anthropic/) will return string content for typical string input:
"""
logger.info("# How to parse text from message objects")


llm = ChatOllama(model="llama3.2")

response = llm.invoke("Hello")
response.content

"""
But when tool calls are generated, the response content is structured into content blocks that convey the model's reasoning process:
"""
logger.info("But when tool calls are generated, the response content is structured into content blocks that convey the model's reasoning process:")



@tool
def get_weather(location: str) -> str:
    """Get the weather from a location."""

    return "Sunny."


llm_with_tools = llm.bind_tools([get_weather])

response = llm_with_tools.invoke("What's the weather in San Francisco, CA?")
response.content

"""
To automatically parse text from message objects irrespective of the format of the underlying content, we can use [StrOutputParser](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html). We can compose it with a chat model as follows:
"""
logger.info("To automatically parse text from message objects irrespective of the format of the underlying content, we can use [StrOutputParser](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html). We can compose it with a chat model as follows:")


chain = llm_with_tools | StrOutputParser()

"""
[StrOutputParser](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html) simplifies the extraction of text from message objects:
"""

response = chain.invoke("What's the weather in San Francisco, CA?")
logger.debug(response)

"""
This is particularly useful in streaming contexts:
"""
logger.info("This is particularly useful in streaming contexts:")

for chunk in chain.stream("What's the weather in San Francisco, CA?"):
    logger.debug(chunk, end="|")

"""
See the [API Reference](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html) for more information.
"""
logger.info("See the [API Reference](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html) for more information.")

logger.info("\n\n[DONE]", bright=True)