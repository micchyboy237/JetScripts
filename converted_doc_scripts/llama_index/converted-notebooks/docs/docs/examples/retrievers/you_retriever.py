from jet.logger import CustomLogger
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.you import YouRetriever
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/retrievers/you_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# You.com Retriever

This notebook walks you through how to setup a Retriever that can fetch from You.com
"""
logger.info("# You.com Retriever")

# %pip install llama-index-retrievers-you


"""
### Retrieve from You.com's Search API
"""
logger.info("### Retrieve from You.com's Search API")

you_api_key = "" or os.environ["YDC_API_KEY"]

retriever = YouRetriever(endpoint="search", api_key=you_api_key)  # default

retrieved_results = retriever.retrieve("national parks in the US")
logger.debug(retrieved_results[0].get_content())

"""
### Retrieve from You.com's News API
"""
logger.info("### Retrieve from You.com's News API")

you_api_key = "" or os.environ["YDC_API_KEY"]

retriever = YouRetriever(endpoint="news", api_key=you_api_key)

retrieved_results = retriever.retrieve("Fed interest rates")
logger.debug(retrieved_results[0].get_content())

"""
## Use in Query Engine
"""
logger.info("## Use in Query Engine")


retriever = YouRetriever()
query_engine = RetrieverQueryEngine.from_args(retriever)

response = query_engine.query("Tell me about national parks in the US")
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)