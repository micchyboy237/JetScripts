from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_exa import ExaFindSimilarResults
from langchain_exa import ExaFindSimilarResults, ExaSearchResults
from langchain_exa import ExaSearchResults
from langchain_exa import ExaSearchRetriever
from langgraph.prebuilt import create_react_agent
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
# Exa Search

Exa is a search engine fully designed for use by LLMs. Search for documents on the internet using **natural language queries**, then retrieve **cleaned HTML content** from desired documents.

Unlike keyword-based search (Google), Exa's neural search capabilities allow it to semantically understand queries and return relevant documents. For example, we could search `"fascinating article about cats"` and compare the search results from [Google](https://www.google.com/search?q=fascinating+article+about+cats) and [Exa](https://search.exa.ai/search?q=fascinating%20article%20about%20cats&autopromptString=Here%20is%20a%20fascinating%20article%20about%20cats%3A). Google gives us SEO-optimized listicles based on the keyword "fascinating". Exa just works.

This notebook goes over how to use Exa Search with LangChain.

## Setup

### Installation

Install the LangChain Exa integration package:
"""
logger.info("# Exa Search")

# %pip install --upgrade --quiet langchain-exa

# %pip install --upgrade --quiet langchain langchain-ollama langchain-community

"""
### Credentials

You'll need an Exa API key to use this integration. Get $10 free credit (plus more by completing certain actions like making your first search) by [signing up here](https://dashboard.exa.ai/).
"""
logger.info("### Credentials")

# import getpass

if not os.environ.get("EXA_API_KEY"):
#     os.environ["EXA_API_KEY"] = getpass.getpass("Exa API key:\n")

"""
## Using ExaSearchResults Tool

ExaSearchResults is a tool that can be used with LangChain agents to perform Exa searches. It provides a more structured interface for search operations:
"""
logger.info("## Using ExaSearchResults Tool")


search_tool = ExaSearchResults(exa_api_key=os.environ["EXA_API_KEY"])

search_results = search_tool._run(
    query="When was the last time the New York Knicks won the NBA Championship?",
    num_results=5,
    text_contents_options=True,
    highlights=True,
)

logger.debug("Search Results:", search_results)

"""
### Advanced Features for ExaSearchResults

You can use advanced search options like controlling search type, live crawling, and content filtering:
"""
logger.info("### Advanced Features for ExaSearchResults")

search_results = search_tool._run(
    query="Latest AI research papers",
    num_results=10,  # Number of results (1-100)
    type="auto",  # Can be "neural", "keyword", or "auto"
    livecrawl="always",  # Can be "always", "fallback", or "never"
    text_contents_options={"max_characters": 2000},  # Limit text length
    summary={"query": "generate one liner"},  # Custom summary prompt
)

logger.debug("Advanced Search Results:")
logger.debug(search_results)

"""
## Using ExaFindSimilarResults Tool

ExaFindSimilarResults allows you to find webpages similar to a given URL. This is useful for finding related content or competitive analysis:
"""
logger.info("## Using ExaFindSimilarResults Tool")


find_similar_tool = ExaFindSimilarResults(exa_api_key=os.environ["EXA_API_KEY"])

similar_results = find_similar_tool._run(
    url="http://espn.com", num_results=5, text_contents_options=True, highlights=True
)

logger.debug("Similar Results:", similar_results)

"""
## Use within an Agent

We can use the ExaSearchResults and ExaFindSimilarResults tools with a LangGraph agent. This gives the agent the ability to dynamically search for information and find similar content based on the user's queries.

First, let's set up the language model. You'll need to provide your Ollama API key:
"""
logger.info("## Use within an Agent")

# import getpass

# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API key:\n")

"""
We will need to install langgraph:
"""
logger.info("We will need to install langgraph:")

# %pip install -qU langgraph


llm = init_chat_model(model="llama3.2", model_provider="ollama", temperature=0)

exa_search = ExaSearchResults(
    exa_api_key=os.environ["EXA_API_KEY"],
    max_results=5,
)

exa_find_similar = ExaFindSimilarResults(
    exa_api_key=os.environ["EXA_API_KEY"],
    max_results=5,
)

agent = create_react_agent(llm, [exa_search, exa_find_similar])

user_input = "What are the latest developments in quantum computing?"

for step in agent.stream(
    {"messages": user_input},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

"""
## Using ExaSearchRetriever

ExaSearchRetriever is a retriever that uses Exa Search to retrieve relevant documents.

:::note

The `max_characters` parameter for **TextContentsOptions** used to be called `max_length` which is now deprecated. Make sure to use `max_characters` instead.

:::

### Basic Usage

Here's a simple example of using ExaSearchRetriever:
"""
logger.info("## Using ExaSearchRetriever")


exa = ExaSearchRetriever(exa_api_key=os.environ["EXA_API_KEY"])

results = exa.invoke("What is the capital of France?")

logger.debug(results)

"""
### Advanced Features

You can use advanced features like controlling the number of results, search type, live crawling, summaries, and text content options:
"""
logger.info("### Advanced Features")


exa = ExaSearchRetriever(
    exa_api_key=os.environ["EXA_API_KEY"],
    k=20,  # Number of results (1-100)
    type="auto",  # Can be "neural", "keyword", or "auto"
    livecrawl="always",  # Can be "always", "fallback", or "never"
    text_contents_options={"max_characters": 3000},  # Limit text length
    summary={"query": "generate one line summary in simple words."},
)

results = exa.invoke("Latest developments in quantum computing")
logger.debug(f"Found {len(results)} results")
for result in results[:3]:  # Print first 3 results
    logger.debug(f"Title: {result.metadata.get('title', 'N/A')}")
    logger.debug(f"URL: {result.metadata.get('url', 'N/A')}")
    logger.debug(f"Summary: {result.metadata.get('summary', 'N/A')}")
    logger.debug("-" * 80)

"""
## API Reference

For detailed documentation of all Exa API features and configurations, visit the [Exa API documentation](https://docs.exa.ai/).
"""
logger.info("## API Reference")

logger.info("\n\n[DONE]", bright=True)