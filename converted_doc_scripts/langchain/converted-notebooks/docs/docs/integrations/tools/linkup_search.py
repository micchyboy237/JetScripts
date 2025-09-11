from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain
from langchain_linkup import LinkupSearchTool
import ChatModelTabs from "@theme/ChatModelTabs";
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
---
sidebar_label: LinkupSearchTool
---

# LinkupSearchTool

> [Linkup](https://www.linkup.so/) provides an API to connect LLMs to the web and the Linkup Premium Partner sources.

This notebook provides a quick overview for getting started with LinkupSearchTool [tool](/docs/concepts/tools/). For detailed documentation of all LinkupSearchTool features and configurations head to the [API reference](https://python.langchain.com/api_reference/linkup/tools/linkup_langchain.search_tool.LinkupSearchTool.html).

## Overview

### Integration details

| Class | Package | Serializable | [JS support](https://js.langchain.com/docs/integrations/tools/linkup_search) |  Package latest |
| :--- | :--- | :---: | :---: | :---: |
| [LinkupSearchTool](https://python.langchain.com/api_reference/linkup/tools/linkup_langchain.search_tool.LinkupSearchTool.html) | [langchain-linkup](https://python.langchain.com/api_reference/linkup/index.html) | ❌ | ❌ |  ![PyPI - Version](https://img.shields.io/pypi/v/langchain-linkup?style=flat-square&label=%20) |

## Setup

To use the Linkup provider, you need a valid API key, which you can find by signing-up [here](https://app.linkup.so/sign-up). To run the following examples you will also need an Ollama API key.

### Installation

This tool lives in the `langchain-linkup` package:
"""
logger.info("# LinkupSearchTool")

# %pip install -qU langchain-linkup

"""
### Credentials
"""
logger.info("### Credentials")

# import getpass

"""
It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability:
"""
logger.info("It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability:")



"""
## Instantiation

Here we show how to instantiate an instance of the LinkupSearchTool tool, with
"""
logger.info("## Instantiation")


tool = LinkupSearchTool(
    depth="deep",  # "standard" or "deep"
    output_type="searchResults",  # "searchResults", "sourcedAnswer" or "structured"
    linkup_api_key=None,  # API key can be passed here or set as the LINKUP_API_KEY environment variable
)

"""
## Invocation

### Invoke directly with args

The tool simply accepts a `query`, which is a string.
"""
logger.info("## Invocation")

tool.invoke({"query": "Who won the latest US presidential elections?"})

"""
### Invoke with ToolCall

We can also invoke the tool with a model-generated ToolCall, in which case a ToolMessage will be returned:
"""
logger.info("### Invoke with ToolCall")

model_generated_tool_call = {
    "args": {"query": "Who won the latest US presidential elections?"},
    "id": "1",
    "name": tool.name,
    "type": "tool_call",
}
tool.invoke(model_generated_tool_call)

"""
## Chaining

We can use our tool in a chain by first binding it to a [tool-calling model](/docs/how_to/tool_calling/) and then calling it:


<ChatModelTabs customVarName="llm" />
"""
logger.info("## Chaining")


llm = init_chat_model(model="llama3.2", model_provider="ollama")


prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{user_input}"),
        ("placeholder", "{messages}"),
    ]
)

llm_with_tools = llm.bind_tools([tool], tool_choice=tool.name)

llm_chain = prompt | llm_with_tools


@chain
def tool_chain(user_input: str, config: RunnableConfig):
    input_ = {"user_input": user_input}
    ai_msg = llm_chain.invoke(input_, config=config)
    tool_msgs = tool.batch(ai_msg.tool_calls, config=config)
    return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)


tool_chain.invoke("Who won the 2016 US presidential elections?")

"""
## API reference

For detailed documentation of all LinkupSearchTool features and configurations head to the [API reference](https://python.langchain.com/api_reference/linkup/tools/linkup_langchain.search_tool.LinkupSearchTool.html).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)