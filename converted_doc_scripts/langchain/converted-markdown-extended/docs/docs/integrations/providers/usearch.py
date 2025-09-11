from jet.logger import logger
from langchain_community.vectorstores import USearch
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
# USearch
>[USearch](https://unum-cloud.github.io/usearch/) is a Smaller & Faster Single-File Vector Search Engine.

>`USearch's` base functionality is identical to `FAISS`, and the interface should look
> familiar if you have ever investigated Approximate Nearest Neighbors search.
> `USearch` and `FAISS` both employ `HNSW` algorithm, but they differ significantly
> in their design principles. `USearch` is compact and broadly compatible with FAISS without
> sacrificing performance, with a primary focus on user-defined metrics and fewer dependencies.
>
## Installation and Setup

We need to install `usearch` python package.
"""
logger.info("# USearch")

pip install usearch

"""
## Vector Store

See a [usage example](/docs/integrations/vectorstores/usearch).
"""
logger.info("## Vector Store")


logger.info("\n\n[DONE]", bright=True)