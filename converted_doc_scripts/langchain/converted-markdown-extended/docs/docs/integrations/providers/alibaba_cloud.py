from jet.logger import logger
from langchain_community.chat_models import PaiEasChatEndpoint
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.document_loaders import MaxComputeLoader
from langchain_community.llms import Tongyi
from langchain_community.llms.pai_eas_endpoint import PaiEasEndpoint
from langchain_community.vectorstores import AlibabaCloudOpenSearch, AlibabaCloudOpenSearchSettings
from langchain_community.vectorstores import AnalyticDB
from langchain_community.vectorstores import Hologres
from langchain_community.vectorstores import TablestoreVectorStore
from langchain_community.vectorstores import Tair
from langchain_qwq import ChatQwQ
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
# Alibaba Cloud

>[Alibaba Group Holding Limited (Wikipedia)](https://en.wikipedia.org/wiki/Alibaba_Group), or `Alibaba`
> (Chinese: 阿里巴巴), is a Chinese multinational technology company specializing in e-commerce, retail,
> Internet, and technology.
>
> [Alibaba Cloud (Wikipedia)](https://en.wikipedia.org/wiki/Alibaba_Cloud), also known as `Aliyun`
> (Chinese: 阿里云; pinyin: Ālǐyún; lit. 'Ali Cloud'), is a cloud computing company, a subsidiary
> of `Alibaba Group`. `Alibaba Cloud` provides cloud computing services to online businesses and
> Alibaba's own e-commerce ecosystem.


## LLMs

### Alibaba Cloud PAI EAS

See [installation instructions and a usage example](/docs/integrations/llms/alibabacloud_pai_eas_endpoint).
"""
logger.info("# Alibaba Cloud")


"""
### Tongyi Qwen

See [installation instructions and a usage example](/docs/integrations/llms/tongyi).
"""
logger.info("### Tongyi Qwen")


"""
## Chat Models

### Alibaba Cloud PAI EAS

See [installation instructions and a usage example](/docs/integrations/chat/alibaba_cloud_pai_eas).
"""
logger.info("## Chat Models")


"""
### Tongyi Qwen Chat

See [installation instructions and a usage example](/docs/integrations/chat/tongyi).
"""
logger.info("### Tongyi Qwen Chat")


"""
### Qwen QwQ Chat

See [installation instructions and a usage example](/docs/integrations/chat/qwq)
"""
logger.info("### Qwen QwQ Chat")


"""
## Document Loaders

### Alibaba Cloud MaxCompute

See [installation instructions and a usage example](/docs/integrations/document_loaders/alibaba_cloud_maxcompute).
"""
logger.info("## Document Loaders")


"""
## Vector stores

### Alibaba Cloud OpenSearch

See [installation instructions and a usage example](/docs/integrations/vectorstores/alibabacloud_opensearch).
"""
logger.info("## Vector stores")


"""
### Alibaba Cloud Tair

See [installation instructions and a usage example](/docs/integrations/vectorstores/tair).
"""
logger.info("### Alibaba Cloud Tair")


"""
### AnalyticDB

See [installation instructions and a usage example](/docs/integrations/vectorstores/analyticdb).
"""
logger.info("### AnalyticDB")


"""
### Hologres

See [installation instructions and a usage example](/docs/integrations/vectorstores/hologres).
"""
logger.info("### Hologres")


"""
### Tablestore

See [installation instructions and a usage example](/docs/integrations/vectorstores/tablestore).
"""
logger.info("### Tablestore")


logger.info("\n\n[DONE]", bright=True)