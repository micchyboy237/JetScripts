from IPython.display import Markdown, display
from jet.logger import CustomLogger
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.vector_stores.myscale import MyScaleVectorStore
from os import environ
import clickhouse_connect
import logging
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/MyScaleIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# MyScale Vector Store
In this notebook we are going to show a quick demo of using the MyScaleVectorStore.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# MyScale Vector Store")

# %pip install llama-index-vector-stores-myscale

# !pip install llama-index

"""
#### Creating a MyScale Client
"""
logger.info("#### Creating a MyScale Client")


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# environ["OPENAI_API_KEY"] = "sk-*"

client = clickhouse_connect.get_client(
    host="YOUR_CLUSTER_HOST",
    port=8443,
    username="YOUR_USERNAME",
    password="YOUR_CLUSTER_PASSWORD",
)

"""
#### Load documents, build and store the VectorStoreIndex with MyScaleVectorStore

Here we will use a set of Paul Graham essays to provide the text to turn into embeddings, store in a ``MyScaleVectorStore`` and query to find context for our LLM QnA loop.
"""
logger.info("#### Load documents, build and store the VectorStoreIndex with MyScaleVectorStore")


documents = SimpleDirectoryReader("./Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()
logger.debug("Document ID:", documents[0].doc_id)
logger.debug("Number of Documents: ", len(documents))

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
You can process your files individually using [SimpleDirectoryReader](/examples/data_connectors/simple_directory_reader.ipynb):
"""
logger.info("You can process your files individually using [SimpleDirectoryReader](/examples/data_connectors/simple_directory_reader.ipynb):")

loader = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/")
documents = loader.load_data()
for file in loader.input_files:
    logger.debug(file)


for document in documents:
    document.metadata = {"user_id": "123", "favorite_color": "blue"}
vector_store = MyScaleVectorStore(myscale_client=client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
#### Query Index

Now MyScale vector store supports filter search and hybrid search

You can learn more about [query_engine](/module_guides/deploying/query_engine/index.md) and [retriever](/module_guides/querying/retriever/index.md).
"""
logger.info("#### Query Index")



query_engine = index.as_query_engine(
    filters=MetadataFilters(
        filters=[
            ExactMatchFilter(key="user_id", value="123"),
        ]
    ),
    similarity_top_k=2,
    vector_store_query_mode="hybrid",
)
response = query_engine.query("What did the author learn?")
logger.debug(textwrap.fill(str(response), 100))

"""
#### Clear All Indexes
"""
logger.info("#### Clear All Indexes")

for document in documents:
    index.delete_ref_doc(document.doc_id)

logger.info("\n\n[DONE]", bright=True)