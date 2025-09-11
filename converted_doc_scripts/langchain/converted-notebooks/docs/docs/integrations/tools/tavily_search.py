from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
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
# Tavily Search

[Tavily's Search API](https://tavily.com) is a search engine built specifically for AI agents (LLMs), delivering real-time, accurate, and factual results at speed.

## Overview

### Integration details
| Class                                                         | Package                                                        | Serializable | [JS support](https://js.langchain.com/docs/integrations/tools/tavily_search) |  Package latest |
|:--------------------------------------------------------------|:---------------------------------------------------------------| :---: | :---: | :---: |
| [TavilySearch](https://github.com/tavily-ai/langchain-tavily) | [langchain-tavily](https://pypi.org/project/langchain-tavily/) | ✅ | ✅  |  ![PyPI - Version](https://img.shields.io/pypi/v/langchain-tavily?style=flat-square&label=%20) |

### Tool features
| [Returns artifact](/docs/how_to/tool_artifacts/) | Native async |                       Return data                        | Pricing |
| :---: | :---: |:--------------------------------------------------------:| :---: |
| ❌ | ✅ | title, URL, content snippet, raw_content, answer, images | 1,000 free searches / month |


## Setup

The integration lives in the `langchain-tavily` package.
"""
logger.info("# Tavily Search")

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

Here we show how to instantiate an instance of the Tavily search tool. The tool accepts various parameters to customize the search. After instantiation we invoke the tool with a simple query. This tool allows you to complete search queries using Tavily's Search API endpoint.

Instantiation
The tool accepts various parameters during instantiation:

- `max_results` (optional, int): Maximum number of search results to return. Default is 5.
- `topic` (optional, str): Category of the search. Can be `'general'`, `'news'`, or `'finance'`. Default is `'general'`.
- `include_answer` (optional, bool): Include an answer to original query in results. Default is False.
- `include_raw_content` (optional, bool): Include cleaned and parsed HTML of each search result. Default is False.
- `include_images` (optional, bool): Include a list of query related images in the response. Default is False.
- `include_image_descriptions` (optional, bool): Include descriptive text for each image. Default is False.
- `search_depth` (optional, str): Depth of the search, either `'basic'` or `'advanced'`. Default is `'basic'`.
- `time_range` (optional, str): The time range back from the current date to filter results - `'day'`, `'week'`, `'month'`, or `'year'`. Default is None.
- `include_domains` (optional, List[str]): List of domains to specifically include. Default is None.
- `exclude_domains` (optional, List[str]): List of domains to specifically exclude. Default is None.

For a comprehensive overview of the available parameters, refer to the [Tavily Search API documentation](https://docs.tavily.com/documentation/api-reference/endpoint/search)
"""
logger.info("## Instantiation")


tool = TavilySearch(
    max_results=5,
    topic="general",
)

"""
## Invocation

### [Invoke directly with args](/docs/concepts/tools)

The Tavily search tool accepts the following arguments during invocation:
- `query` (required): A natural language search query
- The following arguments can also be set during invocation : `include_images`, `search_depth` , `time_range`, `include_domains`, `exclude_domains`, `include_images`
- For reliability and performance reasons, certain parameters that affect response size cannot be modified during invocation: `include_answer` and `include_raw_content`. These limitations prevent unexpected context window issues and ensure consistent results.

:::note

The optional arguments are available for agents to dynamically set, if you set an argument during instantiation and then invoke the tool with a different value, the tool will use the value you passed during invocation.

:::
"""
logger.info("## Invocation")

tool.invoke({"query": "What happened at the last wimbledon"})

"""
### [Invoke with ToolCall](/docs/concepts/tools)

We can also invoke the tool with a model-generated `ToolCall`, in which case a `ToolMessage` will be returned:
"""
logger.info("### [Invoke with ToolCall](/docs/concepts/tools)")

model_generated_tool_call = {
    "args": {"query": "euro 2024 host nation"},
    "id": "1",
    "name": "tavily",
    "type": "tool_call",
}
tool_msg = tool.invoke(model_generated_tool_call)

logger.debug(tool_msg.content[:400])

"""
## Use within an agent

We can use our tools directly with an agent executor by binding the tool to the agent. This gives the agent the ability to dynamically set the available arguments to the Tavily search tool.

In the below example when we ask the agent to find "What nation hosted the Euro 2024? Include only wikipedia sources." the agent will dynamically set the argments and invoke Tavily search tool : Invoking `tavily_search` with `{'query': 'Euro 2024 host nation', 'include_domains': ['wikipedia.org']`


<ChatModelTabs customVarName="llm" />
"""
logger.info("## Use within an agent")

# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("OPENAI_API_KEY:\n")


llm = init_chat_model(model="llama3.2", model_provider="ollama", temperature=0)

"""
We will need to install `langgraph`:
"""
logger.info("We will need to install `langgraph`:")

# %pip install -qU langgraph


tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
)

agent = create_react_agent(llm, [tavily_search_tool])

user_input = "What nation hosted the Euro 2024? Include only wikipedia sources."

for step in agent.stream(
    {"messages": user_input},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

"""
## API reference

For detailed documentation of all Tavily Search API features and configurations head to the [API reference](https://docs.tavily.com/documentation/api-reference/endpoint/search).


"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)