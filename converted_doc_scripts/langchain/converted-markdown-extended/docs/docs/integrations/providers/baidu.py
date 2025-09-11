from jet.logger import logger
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.document_loaders.baiducloud_bos_directory import BaiduBOSDirectoryLoader
from langchain_community.document_loaders.baiducloud_bos_file import BaiduBOSFileLoader
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.llms import QianfanLLMEndpoint
from langchain_community.vectorstores import BESVectorStore
from langchain_community.vectorstores import BaiduVectorDB
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
# Baidu

>[Baidu Cloud](https://cloud.baidu.com/) is a cloud service provided by `Baidu, Inc.`,
> headquartered in Beijing. It offers a cloud storage service, client software,
> file management, resource sharing, and Third Party Integration.


## Installation and Setup

Register and get the `Qianfan` `AK` and `SK` keys [here](https://cloud.baidu.com/product/wenxinworkshop).

## LLMs

### Baidu Qianfan

See a [usage example](/docs/integrations/llms/baidu_qianfan_endpoint).
"""
logger.info("# Baidu")


"""
## Chat models

### Qianfan Chat Endpoint

See a [usage example](/docs/integrations/chat/baidu_qianfan_endpoint).
See another [usage example](/docs/integrations/chat/ernie).
"""
logger.info("## Chat models")


"""
## Embedding models

### Baidu Qianfan

See a [usage example](/docs/integrations/text_embedding/baidu_qianfan_endpoint).
See another [usage example](/docs/integrations/text_embedding/ernie).
"""
logger.info("## Embedding models")


"""
## Document loaders

### Baidu BOS Directory Loader
"""
logger.info("## Document loaders")


"""
### Baidu BOS File Loader
"""
logger.info("### Baidu BOS File Loader")


"""
## Vector stores

### Baidu Cloud ElasticSearch VectorSearch

See a [usage example](/docs/integrations/vectorstores/baiducloud_vector_search).
"""
logger.info("## Vector stores")


"""
### Baidu VectorDB

See a [usage example](/docs/integrations/vectorstores/baiduvectordb).
"""
logger.info("### Baidu VectorDB")


logger.info("\n\n[DONE]", bright=True)