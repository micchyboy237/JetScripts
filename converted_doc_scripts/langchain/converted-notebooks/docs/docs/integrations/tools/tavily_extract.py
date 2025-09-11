from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilyExtract
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
# Tavily Extract

[Tavily](https://tavily.com) is a search engine built specifically for AI agents (LLMs), delivering real-time, accurate, and factual results at speed. Tavily offers an [Extract](https://docs.tavily.com/api-reference/endpoint/extract) endpoint that can be used to extract content from a URLs.


## Overview

### Integration details
| Class                                                         | Package                                                        | Serializable | [JS support](https://js.langchain.com/docs/integrations/tools/tavily_extract/) |  Package latest |
|:--------------------------------------------------------------|:---------------------------------------------------------------| :---: | :---: | :---: |
| [TavilyExtract](https://github.com/tavily-ai/langchain-tavily) | [langchain-tavily](https://pypi.org/project/langchain-tavily/) | ✅ | ✅  |  ![PyPI - Version](https://img.shields.io/pypi/v/langchain-tavily?style=flat-square&label=%20) |

### Tool features
| [Returns artifact](/docs/how_to/tool_artifacts/) | Native async |                       Return data                        | Pricing |
| :---: | :---: |:--------------------------------------------------------:| :---: |
| ❌ | ✅ | raw content and images | 1,000 free searches / month |


## Setup

The integration lives in the `langchain-tavily` package.
"""
logger.info("# Tavily Extract")

# %pip install -qU langchain-tavily

"""
### Credentials

We also need to set our Tavily API key. You can get an API key by visiting [this site](https://app.tavily.com/sign-in) and creating an account.
"""
logger.info("### Credentials")

# import getpass

if not os.environ.get("TAVILY_API_KEY"):
#     os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

"""
## Instantiation


The tool accepts various parameters during instantiation:

- `extract_depth` (optional, str): The depth of the extraction, either `'basic'` or `'advanced'`. Default is `'basic'`.
- `include_images` (optional, bool): Whether to include images in the extraction. Default is False.

For a comprehensive overview of the available parameters, refer to the [Tavily Extract API documentation](https://docs.tavily.com/documentation/api-reference/endpoint/extract)
"""
logger.info("## Instantiation")


tool = TavilyExtract(
    extract_depth="basic",
    include_images=False,
)

"""
## Invocation

### [Invoke directly with args](/docs/concepts/tools)


The Tavily extract tool accepts the following arguments during invocation:
- `urls` (required): A list of URLs to extract content from. 
- Both `extract_depth` and `include_images` can also be set during invocation

:::note

The optional arguments are available for agents to dynamically set, if you set an argument during instantiation and then invoke the tool with a different value, the tool will use the value you passed during invocation.

:::
"""
logger.info("## Invocation")

tool.invoke({"urls": ["https://en.wikipedia.org/wiki/Lionel_Messi"]})

"""
### [Invoke with ToolCall](/docs/concepts/tools)

We can also invoke the tool with a model-generated `ToolCall`, in which case a `ToolMessage` will be returned:
"""
logger.info("### [Invoke with ToolCall](/docs/concepts/tools)")

model_generated_tool_call = {
    "args": {"urls": ["https://en.wikipedia.org/wiki/Lionel_Messi"]},
    "id": "1",
    "name": "tavily",
    "type": "tool_call",
}
tool_msg = tool.invoke(model_generated_tool_call)

logger.debug(tool_msg.content[:400])

"""
## Use within an agent

We can use our tools directly with an agent executor by binding the tool to the agent. This gives the agent the ability to dynamically set the available arguments to the Tavily search tool.


<ChatModelTabs customVarName="llm" />
"""
logger.info("## Use within an agent")

# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("OPENAI_API_KEY:\n")


llm = init_chat_model(model="llama3.2", model_provider="ollama", temperature=0)


tavily_search_tool = TavilyExtract()

agent = create_react_agent(llm, [tavily_search_tool])

user_input = "['https://en.wikipedia.org/wiki/Albert_Einstein','https://en.wikipedia.org/wiki/Theoretical_physics']"

for step in agent.stream(
    {"messages": user_input},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

"""
## API reference

For detailed documentation of all Tavily Search API features and configurations head to the [API reference](https://docs.tavily.com/documentation/api-reference/endpoint/extract).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)