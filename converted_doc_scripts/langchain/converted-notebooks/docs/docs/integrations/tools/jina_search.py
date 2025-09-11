from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_community.tools import JinaSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain
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
sidebar_label: Jina Search
---

# Jina Search

This notebook provides a quick overview for getting started with Jina [tool](/docs/integrations/tools/). For detailed documentation of all Jina features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.jina_search.tool.JinaSearch.html).

## Overview

### Integration details

| Class | Package | Serializable | JS support |  Package latest |
| :--- | :--- | :---: | :---: | :---: |
| [JinaSearch](https://python.langchain.com/api_reference/community/tools/langchain_community.tools.jina_search.tool.JinaSearch.html) | [langchain-community](https://python.langchain.com/api_reference/community/) | ❌ | ❌ |  ![PyPI - Version](https://img.shields.io/pypi/v/langchain-community?style=flat-square&label=%20) |

### Tool features
| [Returns artifact](/docs/how_to/tool_artifacts/) | Native async | Return data | Pricing |
| :---: | :---: | :---: | :---: |
| ❌ | ❌ | URL, Snippet, Title, Page Content | 1M response tokens free | 


## Setup

The integration lives in the `langchain-community` package and was added in version `0.2.16`:
"""
logger.info("# Jina Search")

# %pip install --quiet -U "langchain-community>=0.2.16"

"""
### Credentials
"""
logger.info("### Credentials")

# import getpass

if not os.environ.get("JINA_API_KEY"):
#     os.environ["JINA_API_KEY"] = getpass.getpass("Jina API key:\n")

"""
It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability:
"""
logger.info("It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability:")



"""
## Instantiation

- TODO: Fill in instantiation params

Here we show how to instantiate an instance of the Jina tool, with
"""
logger.info("## Instantiation")


tool = JinaSearch()

"""
## Invocation

### [Invoke directly with args](/docs/concepts/tools)
"""
logger.info("## Invocation")

logger.debug(tool.invoke({"query": "what is langgraph"})[:1000])

"""
### [Invoke with ToolCall](/docs/concepts/tools)

We can also invoke the tool with a model-generated ToolCall, in which case a ToolMessage will be returned:
"""
logger.info("### [Invoke with ToolCall](/docs/concepts/tools)")

model_generated_tool_call = {
    "args": {"query": "what is langgraph"},
    "id": "1",
    "name": tool.name,
    "type": "tool_call",
}
tool_msg = tool.invoke(model_generated_tool_call)
logger.debug(tool_msg.content[:1000])

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


llm_with_tools = llm.bind_tools([tool])
llm_chain = prompt | llm_with_tools


@chain
def tool_chain(user_input: str, config: RunnableConfig):
    input_ = {"user_input": user_input}
    ai_msg = llm_chain.invoke(input_, config=config)
    tool_msgs = tool.batch(ai_msg.tool_calls, config=config)
    return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)


tool_chain.invoke("what's langgraph")

"""
## API reference

For detailed documentation of all Jina features and configurations head to the API reference: https://python.langchain.com/api_reference/community/tools/langchain_community.tools.jina_search.tool.JinaSearch.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)