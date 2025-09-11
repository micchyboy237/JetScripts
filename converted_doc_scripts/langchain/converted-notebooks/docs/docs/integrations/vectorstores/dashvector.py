from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.vectorstores import DashVector
from langchain_text_splitters import CharacterTextSplitter
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
# DashVector

> [DashVector](https://help.aliyun.com/document_detail/2510225.html) is a fully-managed vectorDB service that supports high-dimension dense and sparse vectors, real-time insertion and filtered search. It is built to scale automatically and can adapt to different application requirements.

This notebook shows how to use functionality related to the `DashVector` vector database.

To use DashVector, you must have an API key.
Here are the [installation instructions](https://help.aliyun.com/document_detail/2510223.html).

## Install
"""
logger.info("# DashVector")

# %pip install --upgrade --quiet  langchain-community dashvector dashscope

"""
We want to use `DashScopeEmbeddings` so we also have to get the Dashscope API Key.
"""
logger.info("We want to use `DashScopeEmbeddings` so we also have to get the Dashscope API Key.")

# import getpass

if "DASHVECTOR_API_KEY" not in os.environ:
#     os.environ["DASHVECTOR_API_KEY"] = getpass.getpass("DashVector API Key:")
if "DASHSCOPE_API_KEY" not in os.environ:
#     os.environ["DASHSCOPE_API_KEY"] = getpass.getpass("DashScope API Key:")

"""
## Example
"""
logger.info("## Example")



loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = DashScopeEmbeddings()

"""
We can create DashVector from documents.
"""
logger.info("We can create DashVector from documents.")

dashvector = DashVector.from_documents(docs, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = dashvector.similarity_search(query)
logger.debug(docs)

"""
We can add texts with meta datas and ids, and search with meta filter.
"""
logger.info("We can add texts with meta datas and ids, and search with meta filter.")

texts = ["foo", "bar", "baz"]
metadatas = [{"key": i} for i in range(len(texts))]
ids = ["0", "1", "2"]

dashvector.add_texts(texts, metadatas=metadatas, ids=ids)

docs = dashvector.similarity_search("foo", filter="key = 2")
logger.debug(docs)

"""
### Operating band `partition` parameters

The `partition` parameter defaults to default, and if a non-existent `partition` parameter is passed in, the `partition` will be created automatically.
"""
logger.info("### Operating band `partition` parameters")

texts = ["foo", "bar", "baz"]
metadatas = [{"key": i} for i in range(len(texts))]
ids = ["0", "1", "2"]
partition = "langchain"

dashvector.add_texts(texts, metadatas=metadatas, ids=ids, partition=partition)

query = "What did the president say about Ketanji Brown Jackson"
docs = dashvector.similarity_search(query, partition=partition)

dashvector.delete(ids=ids, partition=partition)

logger.info("\n\n[DONE]", bright=True)