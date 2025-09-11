from jet.logger import logger
from langchain_core.globals import set_llm_cache
from langchain_singlestore import SingleStoreSemanticCache
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
# SingleStoreSemanticCache

This example demonstrates how to get started with the SingleStore semantic cache.

### Integration Overview

`SingleStoreSemanticCache` leverages `SingleStoreVectorStore` to cache LLM responses directly in a SingleStore database, enabling efficient semantic retrieval and reuse of results.

### Integration details



| Class | Package | JS support |
| :--- | :--- |  :---: |
| SingleStoreSemanticCache | langchain_singlestore | ❌ |

## Installation

This cache lives in the `langchain-singlestore` package:
"""
logger.info("# SingleStoreSemanticCache")

# %pip install -qU langchain-singlestore

"""
## Usage
"""
logger.info("## Usage")


set_llm_cache(
    SingleStoreSemanticCache(
        embedding=YourEmbeddings(),
        host="root:pass@localhost:3306/db",
    )
)

# %%time
llm.invoke("Tell me a joke")

# %%time
llm.invoke("Tell me one joke")

logger.info("\n\n[DONE]", bright=True)