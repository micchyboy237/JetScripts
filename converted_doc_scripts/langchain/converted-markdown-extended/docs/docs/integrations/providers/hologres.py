from jet.logger import logger
from langchain_community.vectorstores import Hologres
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
# Hologres

>[Hologres](https://www.alibabacloud.com/help/en/hologres/latest/introduction) is a unified real-time data warehousing service developed by Alibaba Cloud. You can use Hologres to write, update, process, and analyze large amounts of data in real time.
>`Hologres` supports standard `SQL` syntax, is compatible with `PostgreSQL`, and supports most PostgreSQL functions. Hologres supports online analytical processing (OLAP) and ad hoc analysis for up to petabytes of data, and provides high-concurrency and low-latency online data services.

>`Hologres` provides **vector database** functionality by adopting [Proxima](https://www.alibabacloud.com/help/en/hologres/latest/vector-processing).
>`Proxima` is a high-performance software library developed by `Alibaba DAMO Academy`. It allows you to search for the nearest neighbors of vectors. Proxima provides higher stability and performance than similar open-source software such as Faiss. Proxima allows you to search for similar text or image embeddings with high throughput and low latency. Hologres is deeply integrated with Proxima to provide a high-performance vector search service.

## Installation and Setup

Click [here](https://www.alibabacloud.com/zh/product/hologres) to fast deploy a Hologres cloud instance.
"""
logger.info("# Hologres")

pip install hologres-vector

"""
## Vector Store

See a [usage example](/docs/integrations/vectorstores/hologres).
"""
logger.info("## Vector Store")


logger.info("\n\n[DONE]", bright=True)