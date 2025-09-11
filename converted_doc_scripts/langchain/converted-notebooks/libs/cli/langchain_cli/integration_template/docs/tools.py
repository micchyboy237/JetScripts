from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_community.tools import __ModuleName__
from langgraph.prebuilt import create_react_agent
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
sidebar_label: __ModuleName__
---

# __ModuleName__

- TODO: Make sure API reference link is correct.

This notebook provides a quick overview for getting started with __ModuleName__ [tool](/docs/integrations/tools/). For detailed documentation of all __ModuleName__ features and configurations head to the [API reference](https://python.langchain.com/v0.2/api_reference/community/tools/langchain_community.tools.__module_name__.tool.__ModuleName__.html).

- TODO: Add any other relevant links, like information about underlying API, etc.

## Overview

### Integration details

- TODO: Make sure links and features are correct

| Class | Package | Serializable | [JS support](https://js.langchain.com/docs/integrations/tools/__module_name__) |  Package latest |
| :--- | :--- | :---: | :---: | :---: |
| [__ModuleName__](https://python.langchain.com/v0.2/api_reference/community/tools/langchain_community.tools.__module_name__.tool.__ModuleName__.html) | [langchain-community](https://api.python.langchain.com/en/latest/community_api_reference.html) | beta/❌ | ✅/❌ |  ![PyPI - Version](https://img.shields.io/pypi/v/langchain-community?style=flat-square&label=%20) |

### Tool features

- TODO: Add feature table if it makes sense


## Setup

- TODO: Add any additional deps

The integration lives in the `langchain-community` package.
"""
logger.info("# __ModuleName__")

# %pip install --quiet -U langchain-community

"""
### Credentials

- TODO: Add any credentials that are needed
"""
logger.info("### Credentials")

# import getpass

"""
It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability:
"""
logger.info("It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability:")



"""
## Instantiation

- TODO: Fill in instantiation params

Here we show how to instantiate an instance of the __ModuleName__ tool, with
"""
logger.info("## Instantiation")



tool = __ModuleName__(
    ...
)

"""
## Invocation

### [Invoke directly with args](/docs/concepts/tools/#use-the-tool-directly)

- TODO: Describe what the tool args are, fill them in, run cell
"""
logger.info("## Invocation")

tool.invoke({...})

"""
### [Invoke with ToolCall](/docs/concepts/tool_calling/#tool-execution)

We can also invoke the tool with a model-generated ToolCall, in which case a ToolMessage will be returned:

- TODO: Fill in tool args and run cell
"""
logger.info("### [Invoke with ToolCall](/docs/concepts/tool_calling/#tool-execution)")

model_generated_tool_call = {
    "args": {...},  # TODO: FILL IN
    "id": "1",
    "name": tool.name,
    "type": "tool_call",
}
tool.invoke(model_generated_tool_call)

"""
## Use within an agent

- TODO: Add user question and run cells

We can use our tool in an [agent](/docs/concepts/agents/). For this we will need a LLM with [tool-calling](/docs/how_to/tool_calling/) capabilities:


<ChatModelTabs customVarName="llm" />
"""
logger.info("## Use within an agent")


llm = init_chat_model(model="llama3.2", model_provider="ollama")


tools = [tool]
agent = create_react_agent(llm, tools)

example_query = "..."

events = agent.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_logger.debug()

"""
## API reference

For detailed documentation of all __ModuleName__ features and configurations head to the API reference: https://python.langchain.com/v0.2/api_reference/community/tools/langchain_community.tools.__module_name__.tool.__ModuleName__.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)