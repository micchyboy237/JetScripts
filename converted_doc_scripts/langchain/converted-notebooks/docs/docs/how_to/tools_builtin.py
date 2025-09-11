from jet.logger import logger
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
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
---
sidebar_position: 4
sidebar_class_name: hidden
---

# How to use built-in tools and toolkits

:::info Prerequisites

This guide assumes familiarity with the following concepts:

- [LangChain Tools](/docs/concepts/tools)
- [LangChain Toolkits](/docs/concepts/tools)

:::

## Tools

LangChain has a large collection of 3rd party tools. Please visit [Tool Integrations](/docs/integrations/tools/) for a list of the available tools.

:::important

When using 3rd party tools, make sure that you understand how the tool works, what permissions
it has. Read over its documentation and check if anything is required from you
from a security point of view. Please see our [security](https://python.langchain.com/docs/security/) 
guidelines for more information.

:::

Let's try out the [Wikipedia integration](/docs/integrations/tools/wikipedia/).
"""
logger.info("# How to use built-in tools and toolkits")

# !pip install -qU langchain-community wikipedia


api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)

logger.debug(tool.invoke({"query": "langchain"}))

"""
The tool has the following defaults associated with it:
"""
logger.info("The tool has the following defaults associated with it:")

logger.debug(f"Name: {tool.name}")
logger.debug(f"Description: {tool.description}")
logger.debug(f"args schema: {tool.args}")
logger.debug(f"returns directly?: {tool.return_direct}")

"""
## Customizing Default Tools
We can also modify the built in name, description, and JSON schema of the arguments.

When defining the JSON schema of the arguments, it is important that the inputs remain the same as the function, so you shouldn't change that. But you can define custom descriptions for each input easily.
"""
logger.info("## Customizing Default Tools")



class WikiInputs(BaseModel):
    """Inputs to the wikipedia tool."""

    query: str = Field(
        description="query to look up in Wikipedia, should be 3 or less words"
    )


tool = WikipediaQueryRun(
    name="wiki-tool",
    description="look up things in wikipedia",
    args_schema=WikiInputs,
    api_wrapper=api_wrapper,
    return_direct=True,
)

logger.debug(tool.run("langchain"))

logger.debug(f"Name: {tool.name}")
logger.debug(f"Description: {tool.description}")
logger.debug(f"args schema: {tool.args}")
logger.debug(f"returns directly?: {tool.return_direct}")

"""
## How to use built-in toolkits

Toolkits are collections of tools that are designed to be used together for specific tasks. They have convenient loading methods.

All Toolkits expose a `get_tools` method which returns a list of tools.

You're usually meant to use them this way:

```python
# Initialize a toolkit
toolkit = ExampleTookit(...)

# Get list of tools
tools = toolkit.get_tools()
```
"""
logger.info("## How to use built-in toolkits")

logger.info("\n\n[DONE]", bright=True)