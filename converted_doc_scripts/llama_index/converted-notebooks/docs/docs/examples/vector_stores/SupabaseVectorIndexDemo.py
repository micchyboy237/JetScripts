from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader, Document, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.vector_stores.supabase import SupabaseVectorStore
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/SupabaseVectorIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Supabase Vector Store
In this notebook we are going to show how to use [Vecs](https://supabase.github.io/vecs/) to perform vector searches in LlamaIndex.  
See [this guide](https://supabase.github.io/vecs/hosting/) for instructions on hosting a database on Supabase

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Supabase Vector Store")

# %pip install llama-index-vector-stores-supabase

# !pip install llama-index




"""
### Setup OllamaFunctionCalling
The first step is to configure the OllamaFunctionCalling key. It will be used to created embeddings for the documents loaded into the index
"""
logger.info("### Setup OllamaFunctionCalling")


# os.environ["OPENAI_API_KEY"] = "[your_openai_api_key]"

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
### Loading documents
Load the documents stored in the `/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/` using the SimpleDirectoryReader
"""
logger.info("### Loading documents")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()
logger.debug(
    "Document ID:",
    documents[0].doc_id,
    "Document Hash:",
    documents[0].doc_hash,
)

"""
### Create an index backed by Supabase's vector store. 
This will work with all Postgres providers that support pgvector.
If the collection does not exist, we will attempt to create a new collection 

> Note: you need to pass in the embedding dimension if not using OllamaFunctionCalling's text-embedding-ada-002, e.g. `vector_store = SupabaseVectorStore(..., dimension=...)`
"""
logger.info("### Create an index backed by Supabase's vector store.")

vector_store = SupabaseVectorStore(
    postgres_connection_string=(
        "postgresql://<user>:<password>@<host>:<port>/<db_name>"
    ),
    collection_name="base_demo",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
### Query the index
We can now ask questions using our index.
"""
logger.info("### Query the index")

query_engine = index.as_query_engine()
response = query_engine.query("Who is the author?")

logger.debug(textwrap.fill(str(response), 100))

response = query_engine.query("What did the author do growing up?")

logger.debug(textwrap.fill(str(response), 100))

"""
## Using metadata filters
"""
logger.info("## Using metadata filters")


nodes = [
    TextNode(
        **{
            "text": "The Shawshank Redemption",
            "metadata": {
                "author": "Stephen King",
                "theme": "Friendship",
            },
        }
    ),
    TextNode(
        **{
            "text": "The Godfather",
            "metadata": {
                "director": "Francis Ford Coppola",
                "theme": "Mafia",
            },
        }
    ),
    TextNode(
        **{
            "text": "Inception",
            "metadata": {
                "director": "Christopher Nolan",
            },
        }
    ),
]

vector_store = SupabaseVectorStore(
    postgres_connection_string=(
        "postgresql://<user>:<password>@<host>:<port>/<db_name>"
    ),
    collection_name="metadata_filters_demo",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)

"""
Define metadata filters
"""
logger.info("Define metadata filters")


filters = MetadataFilters(
    filters=[ExactMatchFilter(key="theme", value="Mafia")]
)

"""
Retrieve from vector store with filters
"""
logger.info("Retrieve from vector store with filters")

retriever = index.as_retriever(filters=filters)
retriever.retrieve("What is inception about?")

logger.info("\n\n[DONE]", bright=True)