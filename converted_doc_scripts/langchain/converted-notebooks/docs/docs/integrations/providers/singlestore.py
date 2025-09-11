from jet.logger import logger
from langchain_singlestore import (
SingleStoreChatMessageHistory,
SingleStoreLoader,
SingleStoreSemanticCache,
SingleStoreVectorStore,
)
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
# SingleStore Integration

[SingleStore](https://singlestore.com/) is a high-performance, distributed SQL database designed to excel in both [cloud](https://www.singlestore.com/cloud/) and on-premises environments. It offers a versatile feature set, seamless deployment options, and exceptional performance.

This integration provides the following components to leverage SingleStore's capabilities:

- **`SingleStoreLoader`**: Load documents directly from a SingleStore database table.
- **`SingleStoreSemanticCache`**: Use SingleStore as a semantic cache for efficient storage and retrieval of embeddings.
- **`SingleStoreChatMessageHistory`**: Store and retrieve chat message history in SingleStore.
- **`SingleStoreVectorStore`**: Store document embeddings and perform fast vector and full-text searches.

These components enable efficient document storage, embedding management, and advanced search capabilities, combining full-text and vector-based search for fast and accurate queries.
"""
logger.info("# SingleStore Integration")


logger.info("\n\n[DONE]", bright=True)