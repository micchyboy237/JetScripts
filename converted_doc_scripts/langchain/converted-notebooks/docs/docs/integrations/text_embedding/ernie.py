from jet.logger import logger
from langchain_community.embeddings import ErnieEmbeddings
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
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
# ERNIE

[ERNIE Embedding-V1](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/alj562vvu) is a text representation model based on `Baidu Wenxin` large-scale model technology, 
which converts text into a vector form represented by numerical values, and is used in text retrieval, information recommendation, knowledge mining and other scenarios.

**Deprecated Warning**

We recommend users using `langchain_community.embeddings.ErnieEmbeddings` 
to use `langchain_community.embeddings.QianfanEmbeddingsEndpoint` instead.

documentation for `QianfanEmbeddingsEndpoint` is [here](/docs/integrations/text_embedding/baidu_qianfan_endpoint/).

they are 2 why we recommend users to use `QianfanEmbeddingsEndpoint`:

1. `QianfanEmbeddingsEndpoint` support more embedding model in the Qianfan platform.
2. `ErnieEmbeddings` is lack of maintenance and deprecated.

Some tips for migration:
"""
logger.info("# ERNIE")


embeddings = QianfanEmbeddingsEndpoint(
    qianfan_ak="your qianfan ak",
    qianfan_sk="your qianfan sk",
)

"""
## Usage
"""
logger.info("## Usage")


embeddings = ErnieEmbeddings()

query_result = embeddings.embed_query("foo")

doc_results = embeddings.embed_documents(["foo"])

logger.info("\n\n[DONE]", bright=True)