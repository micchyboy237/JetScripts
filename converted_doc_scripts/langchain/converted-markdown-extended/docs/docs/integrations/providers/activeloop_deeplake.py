from jet.logger import logger
from langchain_community.vectorstores import DeepLake
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
# Activeloop Deep Lake

>[Activeloop Deep Lake](https://docs.activeloop.ai/) is a data lake for Deep Learning applications, allowing you to use it
> as a vector store.

## Why Deep Lake?

- More than just a (multi-modal) vector store. You can later use the dataset to fine-tune your own LLM models.
- Not only stores embeddings, but also the original data with automatic version control.
- Truly serverless. Doesn't require another service and can be used with major cloud providers (`AWS S3`, `GCS`, etc.)

`Activeloop Deep Lake` supports `SelfQuery Retrieval`:
[Activeloop Deep Lake Self Query Retrieval](/docs/integrations/retrievers/self_query/activeloop_deeplake_self_query)


## More Resources

1. [Ultimate Guide to LangChain & Deep Lake: Build ChatGPT to Answer Questions on Your Financial Data](https://www.activeloop.ai/resources/ultimate-guide-to-lang-chain-deep-lake-build-chat-gpt-to-answer-questions-on-your-financial-data/)
2. [Twitter the-algorithm codebase analysis with Deep Lake](https://github.com/langchain-ai/langchain/blob/master/cookbook/twitter-the-algorithm-analysis-deeplake.ipynb)
3. Here is [whitepaper](https://www.deeplake.ai/whitepaper) and [academic paper](https://arxiv.org/pdf/2209.10785.pdf) for Deep Lake
4. Here is a set of additional resources available for review: [Deep Lake](https://github.com/activeloopai/deeplake), [Get started](https://docs.activeloop.ai/getting-started) and [Tutorials](https://docs.activeloop.ai/hub-tutorials)

## Installation and Setup

Install the Python package:
"""
logger.info("# Activeloop Deep Lake")

pip install deeplake

"""
## VectorStore
"""
logger.info("## VectorStore")


"""
See a [usage example](/docs/integrations/vectorstores/activeloop_deeplake).
"""
logger.info("See a [usage example](/docs/integrations/vectorstores/activeloop_deeplake).")

logger.info("\n\n[DONE]", bright=True)