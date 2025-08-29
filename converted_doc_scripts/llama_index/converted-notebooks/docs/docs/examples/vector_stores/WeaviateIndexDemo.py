from IPython.display import Markdown, display
from jet.logger import CustomLogger
from llama_index.core import Document
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from weaviate.classes.config import ConsistencyLevel
import logging
import openai
import os
import shutil
import sys
import weaviate


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/WeaviateIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Weaviate Vector Store

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Weaviate Vector Store")

# %pip install llama-index-vector-stores-weaviate

# !pip install llama-index

"""
#### Creating a Weaviate Client
"""
logger.info("#### Creating a Weaviate Client")


# os.environ["OPENAI_API_KEY"] = ""
# openai.api_key = os.environ["OPENAI_API_KEY"]


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


cluster_url = ""
api_key = ""

client = weaviate.connect_to_wcs(
    cluster_url=cluster_url,
    auth_credentials=weaviate.auth.AuthApiKey(api_key),
)

"""
#### Load documents, build the VectorStoreIndex
"""
logger.info("#### Load documents, build the VectorStoreIndex")


"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()


vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name="LlamaIndex"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
#### Using a custom batch configuration

Llamaindex defaults to Weaviate's dynamic batching, optimized for most common scenarios. However, in low-latency setups, this can overload the server or max out any GRPC Message limits in place. For more control and a better ingestion process, consider adjusting batch size by using the fixed size batch.


Here is how you can fine tune WeaviateVectorStore and define a custom batch:
"""
logger.info("#### Using a custom batch configuration")


custom_batch = client.batch.fixed_size(
    batch_size=123,
    concurrent_requests=3,
    consistency_level=ConsistencyLevel.ALL,
)
vector_store_fixed = WeaviateVectorStore(
    weaviate_client=client,
    index_name="LlamaIndex",
    client_kwargs={"custom_batch": custom_batch},
)

"""
#### Query Index
"""
logger.info("#### Query Index")

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

display(Markdown(f"<b>{response}</b>"))

"""
## Loading the index

Here, we use the same index name as when we created the initial index. This stops it from being auto-generated and allows us to easily connect back to it.
"""
logger.info("## Loading the index")

cluster_url = ""
api_key = ""

client = weaviate.connect_to_wcs(
    cluster_url=cluster_url,
    auth_credentials=weaviate.auth.AuthApiKey(api_key),
)

vector_store = WeaviateVectorStore(
    weaviate_client=client, index_name="LlamaIndex"
)

loaded_index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = loaded_index.as_query_engine()
response = query_engine.query("What happened at interleaf?")
display(Markdown(f"<b>{response}</b>"))

"""
## Metadata Filtering

Let's insert a dummy document, and try to filter so that only that document is returned.
"""
logger.info("## Metadata Filtering")


doc = Document.example()
logger.debug(doc.metadata)
logger.debug("-----")
logger.debug(doc.text[:100])

loaded_index.insert(doc)


filters = MetadataFilters(
    filters=[ExactMatchFilter(key="filename", value="README.md")]
)
query_engine = loaded_index.as_query_engine(filters=filters)
response = query_engine.query("What is the name of the file?")
display(Markdown(f"<b>{response}</b>"))

"""
# Deleting the index completely

You can delete the index created by the vector store using the `delete_index` function
"""
logger.info("# Deleting the index completely")

vector_store.delete_index()

vector_store.delete_index()  # calling the function again does nothing

"""
# Connection Termination

You must ensure your client connections are closed:
"""
logger.info("# Connection Termination")

client.close()

logger.info("\n\n[DONE]", bright=True)