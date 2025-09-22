from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_scraperapi.tools import ScraperAPIAmazonSearchTool
from langchain_scraperapi.tools import ScraperAPIGoogleSearchTool
from langchain_scraperapi.tools import ScraperAPITool
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
# LangChain – ScraperAPI

Give your AI agent the ability to browse websites, search Google and Amazon in just two lines of code.

The `langchain-scraperapi` package adds three ready-to-use LangChain tools backed by the [ScraperAPI](https://www.scraperapi.com/) service:

| Tool class | Use it to |
|------------|------------------|
| `ScraperAPITool` | Grab the HTML/text/markdown of any web page |
| `ScraperAPIGoogleSearchTool` | Get structured Google Search SERP data |
| `ScraperAPIAmazonSearchTool` | Get structured Amazon product-search data |

## Overview

### Integration details

| Package | Serializable | [JS support](https://js.langchain.com/docs/integrations/tools/__module_name__) |  Package latest |
| :--- | :---: | :---: | :---: |
| [langchain-scraperapi](https://pypi.org/project/langchain-scraperapi/) | ❌ | ❌ |  v0.1.1 |

### Setup

Install the `langchain-scraperapi` package.
"""
logger.info("# LangChain – ScraperAPI")

# %pip install -U langchain-scraperapi

"""
### Credentials

Create an account at https://www.scraperapi.com/ and get an API key.
"""
logger.info("### Credentials")


os.environ["SCRAPERAPI_API_KEY"] = "your-api-key"

"""
## Instantiation
"""
logger.info("## Instantiation")


tool = ScraperAPITool()

"""
## Invocation
"""
logger.info("## Invocation")

output = tool.invoke(
    {
        "url": "https://langchain.com",
        "output_format": "markdown",
        "render": True,
    }
)
logger.debug(output)

"""
## Features

### 1. `ScraperAPITool` — browse any website

Invoke the *raw* ScraperAPI endpoint and get HTML, rendered DOM, text, or markdown.

**Invocation arguments**

* **`url`** **(required)** – target page URL  
* **Optional (mirror ScraperAPI query params)**  
  * `output_format`: `"text"` | `"markdown"` (default returns raw HTML)  
  * `country_code`: e.g. `"us"`, `"de"`  
  * `device_type`: `"desktop"` | `"mobile"`  
  * `premium`: `bool` – use premium proxies  
  * `render`: `bool` – run JS before returning HTML  
  * `keep_headers`: `bool` – include response headers  
  
For the complete set of modifiers see the [ScraperAPI request-customisation docs](https://docs.scraperapi.com/python/making-requests/customizing-requests)
"""
logger.info("## Features")


tool = ScraperAPITool()

html_text = tool.invoke(
    {
        "url": "https://langchain.com",
        "output_format": "markdown",
        "render": True,
    }
)
logger.debug(html_text[:300], "…")

"""
### 2. `ScraperAPIGoogleSearchTool` — structured Google Search

Structured SERP data via `/structured/google/search`.

**Invocation arguments**

* **`query`** **(required)** – natural-language search string  
* **Optional** — `country_code`, `tld`, `uule`, `hl`, `gl`, `ie`, `oe`, `start`, `num`  
* `output_format`: `"json"` (default) or `"csv"`
"""
logger.info("### 2. `ScraperAPIGoogleSearchTool` — structured Google Search")


google_search = ScraperAPIGoogleSearchTool()

results = google_search.invoke(
    {
        "query": "what is langchain",
        "num": 20,
        "output_format": "json",
    }
)
logger.debug(results)

"""
### 3. `ScraperAPIAmazonSearchTool` — structured Amazon Search

Structured product results via `/structured/amazon/search`.

**Invocation arguments**

* **`query`** **(required)** – product search terms  
* **Optional** — `country_code`, `tld`, `page`  
* `output_format`: `"json"` (default) or `"csv"`
"""
logger.info("### 3. `ScraperAPIAmazonSearchTool` — structured Amazon Search")


amazon_search = ScraperAPIAmazonSearchTool()

products = amazon_search.invoke(
    {
        "query": "noise cancelling headphones",
        "tld": "co.uk",
        "page": 2,
    }
)
logger.debug(products)

"""
## Use within an agent

Here is an example of using the tools in an AI agent. The `ScraperAPITool` gives the AI the ability to browse any website, summarize articles, and click on links to navigate between pages.
"""
logger.info("## Use within an agent")

# %pip install -U langchain-ollama



os.environ["SCRAPERAPI_API_KEY"] = "your-api-key"
# os.environ["OPENAI_API_KEY"] = "your-api-key"

tools = [ScraperAPITool(output_format="markdown")]
llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that can browse websites for users. When asked to browse a website or a link, do so with the ScraperAPITool, then provide information based on the website based on the user's needs.",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
response = agent_executor.invoke(
    {"input": "can you browse hacker news and summarize the first website"}
)

"""
## API reference

Below you can find more information on additional parameters to the tools to customize your requests.

* [ScraperAPITool](https://docs.scraperapi.com/python/making-requests/customizing-requests)
* [ScraperAPIGoogleSearchTool](https://docs.scraperapi.com/python/make-requests-with-scraperapi-in-python/scraperapi-structured-data-collection-in-python/google-serp-api-structured-data-in-python)
* [ScraperAPIAmazonSearchTool](https://docs.scraperapi.com/python/make-requests-with-scraperapi-in-python/scraperapi-structured-data-collection-in-python/amazon-search-api-structured-data-in-python)

The LangChain wrappers surface these parameters directly.
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)