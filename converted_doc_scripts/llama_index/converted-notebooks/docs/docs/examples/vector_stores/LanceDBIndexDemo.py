from datetime import datetime
from jet.logger import CustomLogger
from lancedb.rerankers import ColbertReranker
from llama_index.core import SimpleDirectoryReader, Document, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import (
MetadataFilters,
FilterOperator,
FilterCondition,
MetadataFilter,
)
from llama_index.vector_stores.lancedb import LanceDBVectorStore
import lancedb
import logging
import openai
import os
import shutil
import sys
import textwrap


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/LanceDBIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# LanceDB Vector Store
In this notebook we are going to show how to use [LanceDB](https://www.lancedb.com) to perform vector searches in LlamaIndex

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# LanceDB Vector Store")

# %pip install llama-index llama-index-vector-stores-lancedb

# %pip install lancedb==0.6.13 #Only required if the above cell installs an older version of lancedb (pypi package may not be released yet)

# ! rm -rf ./lancedb





"""
### Setup OllamaFunctionCallingAdapter
The first step is to configure the openai key. It will be used to created embeddings for the documents loaded into the index
"""
logger.info("### Setup OllamaFunctionCallingAdapter")


openai.api_key = "sk-"

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
### Loading documents
Load the documents stored in the `data/paul_graham/` using the SimpleDirectoryReader
"""
logger.info("### Loading documents")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()
logger.debug("Document ID:", documents[0].doc_id, "Document Hash:", documents[0].hash)

"""
### Create the index
Here we create an index backed by LanceDB using the documents loaded previously. LanceDBVectorStore takes a few arguments.
- uri (str, required): Location where LanceDB will store its files.
- table_name (str, optional): The table name where the embeddings will be stored. Defaults to "vectors".
- nprobes (int, optional): The number of probes used. A higher number makes search more accurate but also slower. Defaults to 20.
- refine_factor: (int, optional): Refine the results by reading extra elements and re-ranking them in memory. Defaults to None

- More details can be found at [LanceDB docs](https://lancedb.github.io/lancedb/ann_indexes)

##### For LanceDB cloud :
```python
vector_store = LanceDBVectorStore( 
    uri="db://db_name", # your remote DB URI
    api_key="sk_..", # lancedb cloud api key
    region="your-region" # the region you configured
    ...
)
"""
logger.info("### Create the index")

vector_store = LanceDBVectorStore(
    uri="./lancedb", mode="overwrite", query_type="hybrid"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
### Query the index
We can now ask questions using our index. We can use filtering via `MetadataFilters` or use native lance `where` clause.
"""
logger.info("### Query the index")




query_filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="creation_date",
            operator=FilterOperator.EQ,
            value=datetime.now().strftime("%Y-%m-%d"),
        ),
        MetadataFilter(
            key="file_size", value=75040, operator=FilterOperator.GT
        ),
    ],
    condition=FilterCondition.AND,
)

"""
### Hybrid Search

LanceDB offers hybrid search with reranking capabilities. For complete documentation, refer [here](https://lancedb.github.io/lancedb/hybrid_search/hybrid_search/).

This example uses the `colbert` reranker. The following cell installs the necessary dependencies for `colbert`. If you choose a different reranker, make sure to adjust the dependencies accordingly.
"""
logger.info("### Hybrid Search")

# ! pip install -U torch transformers tantivy@git+https://github.com/quickwit-oss/tantivy-py#164adc87e1a033117001cf70e38c82a53014d985

"""
if you want to add a reranker at vector store initialization, you can pass it in the arguments like below :
```
reranker = ColbertReranker()
vector_store = LanceDBVectorStore(uri="./lancedb", reranker=reranker, mode="overwrite")
```
"""
logger.info("if you want to add a reranker at vector store initialization, you can pass it in the arguments like below :")



reranker = ColbertReranker()
vector_store._add_reranker(reranker)

query_engine = index.as_query_engine(
    filters=query_filters,
)

response = query_engine.query("How much did Viaweb charge per month?")

logger.debug(response)
logger.debug("metadata -", response.metadata)

"""
##### lance filters(SQL like) directly via the `where` clause :
"""
logger.info("##### lance filters(SQL like) directly via the `where` clause :")

lance_filter = "metadata.file_name = 'paul_graham_essay.txt' "
retriever = index.as_retriever(vector_store_kwargs={"where": lance_filter})
response = retriever.retrieve("What did the author do growing up?")

logger.debug(response[0].get_content())
logger.debug("metadata -", response[0].metadata)

"""
### Appending data
You can also add data to an existing index
"""
logger.info("### Appending data")

nodes = [node.node for node in response]

del index

index = VectorStoreIndex.from_documents(
    [Document(text="The sky is purple in Portland, Maine")],
    uri="/tmp/new_dataset",
)

index.insert_nodes(nodes)

query_engine = index.as_query_engine()
response = query_engine.query("Where is the sky purple?")
logger.debug(textwrap.fill(str(response), 100))

"""
You can also create an index from an existing table
"""
logger.info("You can also create an index from an existing table")

del index

vec_store = LanceDBVectorStore.from_table(vector_store._table)
index = VectorStoreIndex.from_vector_store(vec_store)

query_engine = index.as_query_engine()
response = query_engine.query("What companies did the author start?")
logger.debug(textwrap.fill(str(response), 100))

logger.info("\n\n[DONE]", bright=True)