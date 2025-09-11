from jet.logger import logger
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_oceanbase.vectorstores import OceanbaseVectorStore
from pyobvector import ObVecClient
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
---
sidebar_label: Oceanbase
---

# OceanbaseVectorStore

This notebook covers how to get started with the Oceanbase vector store.

## Setup

To access Oceanbase vector stores you'll need to deploy a standalone OceanBase server:

%docker run --name=ob433 -e MODE=mini -e OB_SERVER_IP=127.0.0.1 -p 2881:2881 -d quay.io/oceanbase/oceanbase-ce:4.3.3.1-101000012024102216

And install the `langchain-oceanbase` integration package.

%pip install -qU "langchain-oceanbase"

Check the connection to OceanBase and set the memory usage ratio for vector data:
"""
logger.info("# OceanbaseVectorStore")


tmp_client = ObVecClient()
tmp_client.perform_raw_text_sql("ALTER SYSTEM ob_vector_memory_limit_percentage = 30")

"""
## Initialization

Configure the API key of the embedded model. Here we use `DashScopeEmbeddings` as an example. When deploying `Oceanbase` with a Docker image as described above, simply follow the script below to set the `host`, `port`, `user`, `password`, and `database name`. For other deployment methods, set these parameters according to the actual situation.

%pip install dashscope
"""
logger.info("## Initialization")



DASHSCOPE_API = os.environ.get("DASHSCOPE_API_KEY", "")
connection_args = {
    "host": "127.0.0.1",
    "port": "2881",
    "user": "root@test",
    "password": "",
    "db_name": "test",
}

embeddings = DashScopeEmbeddings(
    model="text-embedding-v1", dashscope_api_key=DASHSCOPE_API
)

vector_store = OceanbaseVectorStore(
    embedding_function=embeddings,
    table_name="langchain_vector",
    connection_args=connection_args,
    vidx_metric_type="l2",
    drop_old=True,
)

"""
## Manage vector store

### Add items to vector store

- TODO: Edit and then run code cell to generate output
"""
logger.info("## Manage vector store")


document_1 = Document(page_content="foo", metadata={"source": "https://foo.com"})
document_2 = Document(page_content="bar", metadata={"source": "https://bar.com"})
document_3 = Document(page_content="baz", metadata={"source": "https://baz.com"})

documents = [document_1, document_2, document_3]

vector_store.add_documents(documents=documents, ids=["1", "2", "3"])

"""
### Update items in vector store
"""
logger.info("### Update items in vector store")

updated_document = Document(
    page_content="qux", metadata={"source": "https://another-example.com"}
)

vector_store.add_documents(documents=[updated_document], ids=["1"])

"""
### Delete items from vector store
"""
logger.info("### Delete items from vector store")

vector_store.delete(ids=["3"])

"""
## Query vector store

Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent. 

### Query directly

Performing a simple similarity search can be done as follows:
"""
logger.info("## Query vector store")

results = vector_store.similarity_search(
    query="thud", k=1, filter={"source": "https://another-example.com"}
)
for doc in results:
    logger.debug(f"* {doc.page_content} [{doc.metadata}]")

"""
If you want to execute a similarity search and receive the corresponding scores you can run:
"""
logger.info("If you want to execute a similarity search and receive the corresponding scores you can run:")

results = vector_store.similarity_search_with_score(
    query="thud", k=1, filter={"source": "https://example.com"}
)
for doc, score in results:
    logger.debug(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

"""
### Query by turning into retriever

You can also transform the vector store into a retriever for easier usage in your chains.
"""
logger.info("### Query by turning into retriever")

retriever = vector_store.as_retriever(search_kwargs={"k": 1})
retriever.invoke("thud")

"""
## Usage for retrieval-augmented generation

For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:

- [Tutorials](/docs/tutorials/)
- [How-to: Question and answer with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
- [Retrieval conceptual docs](https://python.langchain.com/docs/concepts/#retrieval)

## API reference

For detailed documentation of all OceanbaseVectorStore features and configurations head to the API reference: https://python.langchain.com/docs/integrations/vectorstores/oceanbase
"""
logger.info("## Usage for retrieval-augmented generation")

logger.info("\n\n[DONE]", bright=True)