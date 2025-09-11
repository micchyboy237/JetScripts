from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain
from langchain_scrapegraph.tools import (
GetCreditsTool,
MarkdownifyTool,
SmartCrawlerTool,
SmartScraperTool,
)
from langchain_scrapegraph.tools import SmartCrawlerTool
from scrapegraph_py.logger import sgai_logger
import ChatModelTabs from "@theme/ChatModelTabs";
import json
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
sidebar_label: ScrapeGraph
---

# ScrapeGraph

This notebook provides a quick overview for getting started with ScrapeGraph [tools](/docs/integrations/tools/). For detailed documentation of all ScrapeGraph features and configurations head to the [API reference](https://python.langchain.com/docs/integrations/tools/scrapegraph).

For more information about ScrapeGraph AI:
- [ScrapeGraph AI Website](https://scrapegraphai.com)
- [Open Source Project](https://github.com/ScrapeGraphAI/Scrapegraph-ai)

## Overview

### Integration details

| Class | Package | Serializable | JS support | Package latest |
| :--- | :--- | :---: | :---: | :---: |
| [SmartScraperTool](https://python.langchain.com/docs/integrations/tools/scrapegraph) | langchain-scrapegraph | ✅ | ❌ | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-scrapegraph?style=flat-square&label=%20) |
| [SmartCrawlerTool](https://python.langchain.com/docs/integrations/tools/scrapegraph) | langchain-scrapegraph | ✅ | ❌ | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-scrapegraph?style=flat-square&label=%20) |
| [MarkdownifyTool](https://python.langchain.com/docs/integrations/tools/scrapegraph) | langchain-scrapegraph | ✅ | ❌ | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-scrapegraph?style=flat-square&label=%20) |
| [GetCreditsTool](https://python.langchain.com/docs/integrations/tools/scrapegraph) | langchain-scrapegraph | ✅ | ❌ | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-scrapegraph?style=flat-square&label=%20) |

### Tool features

| Tool | Purpose | Input | Output |
| :--- | :--- | :--- | :--- |
| SmartScraperTool | Extract structured data from websites | URL + prompt | JSON |
| SmartCrawlerTool | Extract data from multiple pages with crawling | URL + prompt + crawl options | JSON |
| MarkdownifyTool | Convert webpages to markdown | URL | Markdown text |
| GetCreditsTool | Check API credits | None | Credit info |


## Setup

The integration requires the following packages:
"""
logger.info("# ScrapeGraph")

# %pip install --quiet -U langchain-scrapegraph

"""
### Credentials

You'll need a ScrapeGraph AI API key to use these tools. Get one at [scrapegraphai.com](https://scrapegraphai.com).
"""
logger.info("### Credentials")

# import getpass

if not os.environ.get("SGAI_API_KEY"):
#     os.environ["SGAI_API_KEY"] = getpass.getpass("ScrapeGraph AI API key:\n")

"""
It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability:
"""
logger.info("It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability:")

os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()

"""
## Instantiation

Here we show how to instantiate instances of the ScrapeGraph tools:
"""
logger.info("## Instantiation")



sgai_logger.set_logging(level="INFO")

smartscraper = SmartScraperTool()
smartcrawler = SmartCrawlerTool()
markdownify = MarkdownifyTool()
credits = GetCreditsTool()

"""
## Invocation

### [Invoke directly with args](/docs/concepts/tools)

Let's try each tool individually:

### SmartCrawler Tool

The SmartCrawlerTool allows you to crawl multiple pages from a website and extract structured data with advanced crawling options like depth control, page limits, and domain restrictions.
"""
logger.info("## Invocation")

result = smartscraper.invoke(
    {
        "user_prompt": "Extract the company name and description",
        "website_url": "https://scrapegraphai.com",
    }
)
logger.debug("SmartScraper Result:", result)

markdown = markdownify.invoke({"website_url": "https://scrapegraphai.com"})
logger.debug("\nMarkdownify Result (first 200 chars):", markdown[:200])

url = "https://scrapegraphai.com/"
prompt = (
    "What does the company do? and I need text content from their privacy and terms"
)

result_crawler = smartcrawler.invoke(
    {
        "url": url,
        "prompt": prompt,
        "cache_website": True,
        "depth": 2,
        "max_pages": 2,
        "same_domain_only": True,
    }
)

logger.debug("\nSmartCrawler Result:")
logger.debug(json.dumps(result_crawler, indent=2))

credits_info = credits.invoke({})
logger.debug("\nCredits Info:", credits_info)



sgai_logger.set_logging(level="INFO")

tool = SmartCrawlerTool()

url = "https://scrapegraphai.com/"
prompt = (
    "What does the company do? and I need text content from their privacy and terms"
)

result = tool.invoke(
    {
        "url": url,
        "prompt": prompt,
        "cache_website": True,
        "depth": 2,
        "max_pages": 2,
        "same_domain_only": True,
    }
)

logger.debug(json.dumps(result, indent=2))

"""
### [Invoke with ToolCall](/docs/concepts/tools)

We can also invoke the tool with a model-generated ToolCall:
"""
logger.info("### [Invoke with ToolCall](/docs/concepts/tools)")

model_generated_tool_call = {
    "args": {
        "user_prompt": "Extract the main heading and description",
        "website_url": "https://scrapegraphai.com",
    },
    "id": "1",
    "name": smartscraper.name,
    "type": "tool_call",
}
smartscraper.invoke(model_generated_tool_call)

"""
## Chaining

Let's use our tools with an LLM to analyze a website:


<ChatModelTabs customVarName="llm" />
"""
logger.info("## Chaining")


llm = init_chat_model(model="llama3.2", model_provider="ollama")


prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistant that can use tools to extract structured information from websites.",
        ),
        ("human", "{user_input}"),
        ("placeholder", "{messages}"),
    ]
)

llm_with_tools = llm.bind_tools([smartscraper], tool_choice=smartscraper.name)
llm_chain = prompt | llm_with_tools


@chain
def tool_chain(user_input: str, config: RunnableConfig):
    input_ = {"user_input": user_input}
    ai_msg = llm_chain.invoke(input_, config=config)
    tool_msgs = smartscraper.batch(ai_msg.tool_calls, config=config)
    return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)


tool_chain.invoke(
    "What does ScrapeGraph AI do? Extract this information from their website https://scrapegraphai.com"
)

"""
## API reference

For detailed documentation of all ScrapeGraph features and configurations head to [the Langchain API reference](https://python.langchain.com/docs/integrations/tools/scrapegraph).

Or to [the official SDK repo](https://github.com/ScrapeGraphAI/langchain-scrapegraph).


"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)