from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
import shutil
import {ItemTable} from "@theme/FeatureTables";


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
sidebar_label: TavilySearchAPI
---

# TavilySearchAPIRetriever

>[Tavily's Search API](https://tavily.com) is a search engine built specifically for AI agents (LLMs), delivering real-time, accurate, and factual results at speed.

We can use this as a [retriever](/docs/how_to#retrievers). It will show functionality specific to this integration. After going through, it may be useful to explore [relevant use-case pages](/docs/how_to#qa-with-rag) to learn how to use this vectorstore as part of a larger chain.

### Integration details


<ItemTable category="external_retrievers" item="TavilySearchAPIRetriever" />

## Setup

If you want to get automated tracing from individual queries, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("# TavilySearchAPIRetriever")



"""
### Installation

The integration lives in the `langchain-community` package. We also need to install the `tavily-python` package itself.
"""
logger.info("### Installation")

# %pip install -qU langchain-community tavily-python

"""
We also need to set our Tavily API key.
"""
logger.info("We also need to set our Tavily API key.")

# import getpass

# os.environ["TAVILY_API_KEY"] = getpass.getpass()

"""
## Instantiation

Now we can instantiate our retriever:
"""
logger.info("## Instantiation")


retriever = TavilySearchAPIRetriever(k=3)

"""
## Usage
"""
logger.info("## Usage")

query = "what year was breath of the wild released?"

retriever.invoke(query)

"""
## Use within a chain

We can easily combine this retriever in to a chain.
"""
logger.info("## Use within a chain")


prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)

llm = ChatOllama(model="llama3.2")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

chain.invoke("how many units did bretch of the wild sell in 2020")

"""
## API reference

For detailed documentation of all `TavilySearchAPIRetriever` features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/retrievers/langchain_community.retrievers.tavily_search_api.TavilySearchAPIRetriever.html).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)