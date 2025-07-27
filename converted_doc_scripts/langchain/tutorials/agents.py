from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.features.search_and_chat import compare_html_query_scores
from jet.file.utils import save_file
from jet.llm.models import OLLAMA_EMBED_MODELS
from jet.llm.ollama.constants import OLLAMA_LARGE_EMBED_MODEL
from jet.llm.query.retrievers import setup_index
from jet.scrapers.browser.playwright_utils import scrape_multiple_urls, ascrape_multiple_urls
from jet.scrapers.utils import safe_path_from_url, search_data, validate_headers
from jet.search.searxng import search_searxng
from jet.token.token_utils import filter_texts
from jet.utils.url_utils import normalize_url
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import Document as LlamaDocument
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
import asyncio
from jet.llm.ollama.base import initialize_ollama_settings
from jet.llm.ollama.base_langchain import ChatOllama
from jet.transformers.formatters import format_json
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import os
import os
from langchain_community.tools.tavily_search import TavilySearchResults
# import ChatModelTabs from "@theme/ChatModelTabs";
from jet.llm.ollama.base_langchain import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from jet.logger import CustomLogger


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


def setup_logger():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(
        script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    return logger


logger = setup_logger()
initialize_ollama_settings()


# @tool
# def search(query: str, config: RunnableConfig) -> list[str]:
#     """
#     A search engine optimized for comprehensive, accurate, and trusted results.
#     Useful for when you need to answer questions about current events.
#     Input should be a search query.
#     """

#     results = search_searxng(
#         query_url="http://Jethros-MacBook-Air.local:3000/search",
#         query=query,
#         min_score=0,
#         engines=["google"],
#         config={
#             "port": 3101
#         },
#     )

#     documents = [LlamaDocument(text=result['content']) for result in results]

#     query_nodes = setup_index(documents, embed_model=OLLAMA_LARGE_EMBED_MODEL)

#     logger.newline()
#     result = query_nodes(query, threshold=0.3,
#                          fusion_mode=FUSION_MODES.RELATIVE_SCORE)

#     return result['texts']

async def aget_url_html_tuples(urls: list[str], top_n: int = 3, min_header_count: int = 5) -> list[tuple[str, str]]:
    url_html_tuples = []
    async for url, html in ascrape_multiple_urls(urls, top_n=top_n, num_parallel=3):
        if html and validate_headers(html, min_header_count):
            url_html_tuples.append((url, html))
            logger.orange(
                f"Scraped urls count: {len(url_html_tuples)} / {top_n}")
            if len(url_html_tuples) == top_n:
                logger.success(
                    f"Scraped urls ({len(url_html_tuples)}) now match {top_n}")
                break
    logger.success(f"Done scraping urls {len(url_html_tuples)}")
    for url, html in url_html_tuples:
        sub_dir = os.path.join(OUTPUT_DIR, "searched_html")
        output_dir_url = safe_path_from_url(url, sub_dir)
        os.makedirs(output_dir_url, exist_ok=True)

        save_file(html, os.path.join(output_dir_url, "doc.html"))
        save_file("\n\n".join([header["content"] for header in get_md_header_contents(
            html)]), os.path.join(output_dir_url, "doc.md"))
    return url_html_tuples


def get_url_html_tuples(urls: list[str], top_n: int = 3, min_header_count: int = 5) -> list[tuple[str, str]]:
    url_html_tuples = []
    for url, html in scrape_multiple_urls(urls, top_n=top_n, num_parallel=3):
        if html and validate_headers(html, min_header_count):
            url_html_tuples.append((url, html))
            logger.orange(
                f"Scraped urls count: {len(url_html_tuples)} / {top_n}")
            if len(url_html_tuples) == top_n:
                logger.success(
                    f"Scraped urls ({len(url_html_tuples)}) now match {top_n}")
                break
    logger.success(f"Done scraping urls {len(url_html_tuples)}")
    for url, html in url_html_tuples:
        sub_dir = os.path.join(OUTPUT_DIR, "searched_html")
        output_dir_url = safe_path_from_url(url, sub_dir)
        os.makedirs(output_dir_url, exist_ok=True)

        save_file(html, os.path.join(output_dir_url, "doc.html"))
        save_file("\n\n".join([header["content"] for header in get_md_header_contents(
            html)]), os.path.join(output_dir_url, "doc.md"))
    return url_html_tuples


@tool
def search(query: str, config: RunnableConfig) -> list[str]:
    """
    A search engine optimized for comprehensive, accurate, and trusted results.
    Useful for when you need to answer questions about current events.
    Input should be a search query.
    """
    logger.info(f"Starting search for query: {query}")

    embed_models: list[OLLAMA_EMBED_MODELS] = [
        "all-minilm:33m", "paraphrase-multilingual"]

    try:
        # Search urls
        logger.debug("Calling search_data")
        search_results = search_data(query)
        logger.debug(f"Search results: {len(search_results)}")

        logger.debug("Running get_url_html_tuples")
        urls = [normalize_url(item["url"]) for item in search_results]
        # url_html_tuples = asyncio.run(aget_url_html_tuples(urls))
        url_html_tuples = get_url_html_tuples(urls)
        logger.success(
            f"Done scraping urls {len(url_html_tuples)} for query: {query}")

        logger.debug(
            f"Retrieved {len(search_results)} search results and {len(url_html_tuples)} URL-HTML tuples")

        comparison_results = compare_html_query_scores(
            query, url_html_tuples, embed_models)

        top_urls = comparison_results["top_urls"]
        top_query_scores = comparison_results["top_query_scores"]
        top_texts = [result["text"] for result in top_query_scores]
        filtered_top_texts: list[str] = filter_texts(
            top_texts, model="llama3.1", max_tokens=1000)

        logger.info(f"Returning {len(filtered_top_texts)} filtered texts")
        return filtered_top_texts

    except Exception as e:
        logger.error(f"Error in search function: {str(e)}", exc_info=True)
        raise


"""
---
keywords: [agent, agents]
---
"""

"""
# Build an Agent

By themselves, language models can't take actions - they just output text.
A big use case for LangChain is creating **agents**.
[Agents](/docs/concepts/agents) are systems that use [LLMs](/docs/concepts/chat_models) as reasoning engines to determine which actions to take and the inputs necessary to perform the action.
After executing actions, the results can be fed back into the LLM to determine whether more actions are needed, or whether it is okay to finish. This is often achieved via [tool-calling](/docs/concepts/tool_calling).

In this tutorial we will build an agent that can interact with a search engine. You will be able to ask this agent questions, watch it call the search tool, and have conversations with it.

## End-to-end agent

The code snippet below represents a fully functional agent that uses an LLM to decide which tools to use. It is equipped with a generic search tool. It has conversational memory - meaning that it can be used as a multi-turn chatbot.

In the rest of the guide, we will walk through the individual components and what each part does - but if you want to just grab some code and get started, feel free to use this!
"""

logger.orange("\n----\nExamples 1\n----\n")

memory = MemorySaver()
model = ChatOllama(model="llama3.1")
# search = TavilySearchResults(max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

logger.info("Query 1 stream response:")
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]}, config
):
    logger.newline()
    logger.success(format_json(chunk))
    logger.gray("----")

logger.info("Query 2 stream response:")
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(
        content="whats the weather where I live?")]}, config
):
    logger.newline()
    logger.success(format_json(chunk))
    logger.gray("----")

"""
## Setup

### Jupyter Notebook

This guide (and most of the other guides in the documentation) uses [Jupyter notebooks](https://jupyter.org/) and assumes the reader is as well. Jupyter notebooks are perfect interactive environments for learning how to work with LLM systems because oftentimes things can go wrong (unexpected output, API down, etc), and observing these cases is a great way to better understand building with LLMs.

This and other tutorials are perhaps most conveniently run in a Jupyter notebook. See [here](https://jupyter.org/install) for instructions on how to install.

### Installation

To install LangChain run:
"""

# %pip install -U langchain-community langgraph langchain-anthropic tavily-python langgraph-checkpoint-sqlite

"""
For more details, see our [Installation guide](/docs/how_to/installation).

### LangSmith

Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls.
As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent.
The best way to do this is with [LangSmith](https://smith.langchain.com).

After you sign up at the link above, make sure to set your environment variables to start logging traces:

```shell
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```

Or, if in a notebook, you can set them with:

```python
# import getpass

os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

### Tavily

We will be using [Tavily](/docs/integrations/tools/tavily_search) (a search engine) as a tool.
In order to use it, you will need to get and set an API key:

```bash
export TAVILY_API_KEY="..."
```

Or, if in a notebook, you can set it with:

```python
# import getpass

# os.environ["TAVILY_API_KEY"] = getpass.getpass()
```
"""

"""
## Define tools

We first need to create the tools we want to use. Our main tool of choice will be [Tavily](/docs/integrations/tools/tavily_search) - a search engine. We have a built-in tool in LangChain to easily use Tavily search engine as tool.
"""


# search = TavilySearchResults(max_results=2)
# search_results = search.invoke("what is the weather in SF")
# logger.debug(search_results)

logger.orange("\n----\nExamples 2\n----\n")

tools = [search]

"""
## Using Language Models

Next, let's learn how to use a language model to call tools. LangChain supports many different language models that you can use interchangably - select the one you want to use below!


<ChatModelTabs openaiParams={`model="llama3.1", request_timeout=300.0, context_window=4096`} />
"""


model = ChatOllama(model="llama3.1")

"""
You can call the language model by passing in a list of messages. By default, the response is a `content` string.
"""


response = model.invoke([HumanMessage(content="hi!")])
response.content

"""
We can now see what it is like to enable this model to do tool calling. In order to enable that we use `.bind_tools` to give the language model knowledge of these tools
"""

model_with_tools = model.bind_tools(tools)

"""
We can now call the model. Let's first call it with a normal message, and see how it responds. We can look at both the `content` field as well as the `tool_calls` field.
"""

response = model_with_tools.invoke([HumanMessage(content="Hi!")])

logger.newline()
logger.info("Query 1 tool response:")
logger.log("ContentString:", f"{
           response.content}", colors=["DEBUG", "SUCCESS"])
logger.log("ToolCalls:", f"{response.tool_calls}", colors=["DEBUG", "SUCCESS"])

"""
Now, let's try calling it with some input that would expect a tool to be called.
"""

response = model_with_tools.invoke(
    [HumanMessage(content="What's the weather in SF?")])

logger.newline()
logger.info("Query 2 tool response:")
logger.log("ContentString:", f"{
           response.content}", colors=["DEBUG", "SUCCESS"])
logger.log("ToolCalls:", f"{response.tool_calls}", colors=["DEBUG", "SUCCESS"])

"""
We can see that there's now no text content, but there is a tool call! It wants us to call the Tavily Search tool.

This isn't calling that tool yet - it's just telling us to. In order to actually call it, we'll want to create our agent.
"""

"""
## Create the agent

Now that we have defined the tools and the LLM, we can create the agent. We will be using [LangGraph](/docs/concepts/architecture/#langgraph) to construct the agent. 
Currently, we are using a high level interface to construct the agent, but the nice thing about LangGraph is that this high-level interface is backed by a low-level, highly controllable API in case you want to modify the agent logic.
"""

"""
Now, we can initialize the agent with the LLM and the tools.

Note that we are passing in the `model`, not `model_with_tools`. That is because `create_react_agent` will call `.bind_tools` for us under the hood.
"""

logger.orange("\n----\nExamples 3\n----\n")

agent_executor = create_react_agent(model, tools)

"""
## Run the agent

We can now run the agent with a few queries! Note that for now, these are all **stateless** queries (it won't remember previous interactions). Note that the agent will return the **final** state at the end of the interaction (which includes any inputs, we will see later on how to get only the outputs).

First up, let's see how it responds when there's no need to call a tool:
"""

response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})

# response["messages"]
logger.newline()
logger.info("Query 1 react agent tool response:")
logger.success(format_json(response["messages"]))

"""
In order to see exactly what is happening under the hood (and to make sure it's not calling a tool) we can take a look at the [LangSmith trace](https://smith.langchain.com/public/28311faa-e135-4d6a-ab6b-caecf6482aaa/r)

Let's now try it out on an example where it should be invoking the tool
"""

response = agent_executor.invoke(
    {"messages": [HumanMessage(content="whats the weather in sf?")]}
)
# response["messages"]
logger.newline()
logger.info("Query 2 react agent tool response:")
logger.success(format_json(response["messages"]))

"""
We can check out the [LangSmith trace](https://smith.langchain.com/public/f520839d-cd4d-4495-8764-e32b548e235d/r) to make sure it's calling the search tool effectively.
"""

"""
## Streaming Messages

We've seen how the agent can be called with `.invoke` to get  a final response. If the agent executes multiple steps, this may take a while. To show intermediate progress, we can stream back messages as they occur.
"""

logger.orange("\n----\nExamples 4\n----\n")

logger.newline()
logger.info("Query 1 react agent stream response:")
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather in sf?")]}
):
    logger.newline()
    logger.success(format_json(chunk))
    logger.gray("----")

"""
## Streaming tokens

In addition to streaming back messages, it is also useful to stream back tokens.
We can do this with the `.astream_events` method.

:::important
This `.astream_events` method only works with Python 3.11 or higher.
:::
"""

logger.newline()
logger.info("Query 2 react agent stream response:")


async def run_inline():
    async for event in agent_executor.astream_events(
        {"messages": [HumanMessage(content="whats the weather in sf?")]}, version="v1"
    ):
        kind = event["event"]
        if kind == "on_chain_start":
            if (
                event["name"] == "Agent"
                # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
            ):
                logger.debug(
                    f"Starting agent: {event['name']} with input: {
                        event['data'].get('input')}"
                )
        elif kind == "on_chain_end":
            if (
                event["name"] == "Agent"
                # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
            ):
                logger.debug()
                logger.debug("--")
                logger.debug(
                    f"Done agent: {event['name']} with output: {
                        event['data'].get('output')['output']}"
                )
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                logger.debug(content, end="|")
        elif kind == "on_tool_start":
            logger.debug("--")
            logger.debug(
                f"Starting tool: {event['name']} with inputs:"
            )
            logger.info(event['data'].get('input'))
        elif kind == "on_tool_end":
            logger.debug(f"Done tool: {event['name']}")
            logger.debug(f"Tool output was:")
            logger.success(event['data'].get('output'))
            logger.debug("--")
asyncio.run(run_inline())

"""
## Adding in memory

As mentioned earlier, this agent is stateless. This means it does not remember previous interactions. To give it memory we need to pass in a checkpointer. When passing in a checkpointer, we also have to pass in a `thread_id` when invoking the agent (so it knows which thread/conversation to resume from).
"""

logger.orange("\n----\nExamples 5\n----\n")

config = {"configurable": {"thread_id": "abc123"}}
model = ChatOllama(model="llama3.1")
tools = []
memory = MemorySaver()
agent_executor = create_react_agent(model, tools, checkpointer=memory)

logger.newline()
logger.info("Query 1 stream response:")
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob!")]}, config
):
    logger.newline()
    logger.success(format_json(chunk))
    logger.gray("----")

logger.info("Query 2 stream response:")
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats my name?")]}, config
):
    logger.newline()
    logger.success(format_json(chunk))
    logger.gray("----")

"""
Example [LangSmith trace](https://smith.langchain.com/public/fa73960b-0f7d-4910-b73d-757a12f33b2b/r)
"""

"""
If you want to start a new conversation, all you have to do is change the `thread_id` used
"""

logger.orange("\n----\nExamples 6\n----\n")

config = {"configurable": {"thread_id": "xyz123"}}
model = ChatOllama(model="llama3.1")
tools = []
memory = MemorySaver()
agent_executor = create_react_agent(model, tools, checkpointer=memory)

logger.info("Query 1 stream response:")
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats my name?")]}, config
):
    logger.newline()
    logger.success(format_json(chunk))
    logger.gray("----")

"""
## Conclusion

That's a wrap! In this quick start we covered how to create a simple agent. 
We've then shown how to stream back a response - not only with the intermediate steps, but also tokens!
We've also added in memory so you can have a conversation with them.
Agents are a complex topic with lots to learn! 

For more information on Agents, please check out the [LangGraph](/docs/concepts/architecture/#langgraph) documentation. This has it's own set of concepts, tutorials, and how-to guides.
"""


logger.info("\n\n[DONE]", bright=True)
