from jet.logger import logger
from langchain_community.retrievers.llama_index import LlamaIndexGraphRetriever
from langchain_community.retrievers.llama_index import LlamaIndexRetriever
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
# LlamaIndex

>[LlamaIndex](https://www.llamaindex.ai/) is the leading data framework for building LLM applications


## Installation and Setup

You need to install the `llama-index` python package.
"""
logger.info("# LlamaIndex")

pip install llama-index

"""
See the [installation instructions](https://docs.llamaindex.ai/en/stable/getting_started/installation/).

## Retrievers

### LlamaIndexRetriever

>It is used for the question-answering with sources over an LlamaIndex data structure.
"""
logger.info("## Retrievers")


"""
### LlamaIndexGraphRetriever

>It is used for question-answering with sources over an LlamaIndex graph data structure.
"""
logger.info("### LlamaIndexGraphRetriever")


logger.info("\n\n[DONE]", bright=True)