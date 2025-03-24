from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.you import YouRetriever
import os
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/retrievers/you_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# You.com Retriever
#
# This notebook walks you through how to setup a Retriever that can fetch from You.com

# %pip install llama-index-retrievers-you


# Retrieve from You.com's Search API

you_api_key = "" or os.environ["YDC_API_KEY"]

retriever = YouRetriever(endpoint="search", api_key=you_api_key)  # default

retrieved_results = retriever.retrieve("national parks in the US")
print(retrieved_results[0].get_content())

# Retrieve from You.com's News API

you_api_key = "" or os.environ["YDC_API_KEY"]

retriever = YouRetriever(endpoint="news", api_key=you_api_key)

retrieved_results = retriever.retrieve("Fed interest rates")
print(retrieved_results[0].get_content())

# Use in Query Engine


retriever = YouRetriever()
query_engine = RetrieverQueryEngine.from_args(retriever)

response = query_engine.query("Tell me about national parks in the US")
print(str(response))

logger.info("\n\n[DONE]", bright=True)
