from jet.transformers.formatters import format_json
from IPython.display import Markdown, display
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
import logging
import os
import qdrant_client
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/QdrantIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Qdrant Vector Store

#### Creating a Qdrant client
"""
logger.info("# Qdrant Vector Store")

# %pip install llama-index-vector-stores-qdrant llama-index-readers-file llama-index-embeddings-fastembed llama-index-llms-ollama



Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")

"""
If running for the first, time, install the dependencies using:

```
!pip install -U qdrant_client fastembed
```

Set your OllamaFunctionCalling key for authenticating the LLM

# Follow these set the OllamaFunctionCalling API key to the OPENAI_API_KEY environment variable -

1. Using Terminal
"""
logger.info("If running for the first, time, install the dependencies using:")

# export OPENAI_API_KEY=your_api_key_here

"""
2. Using IPython Magic Command in Jupyter Notebook
"""
logger.info("2. Using IPython Magic Command in Jupyter Notebook")

# %env OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>

"""
3. Using Python Script
"""
logger.info("3. Using Python Script")


# os.environ["OPENAI_API_KEY"] = "your_api_key_here"

"""
Note: It's generally recommended to set sensitive information like API keys as environment variables rather than hardcoding them into scripts.
"""
logger.info("Note: It's generally recommended to set sensitive information like API keys as environment variables rather than hardcoding them into scripts.")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
#### Load the documents
"""
logger.info("#### Load the documents")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

"""
#### Build the VectorStoreIndex
"""
logger.info("#### Build the VectorStoreIndex")

client = qdrant_client.QdrantClient(
    host="localhost",
    port=6333
)

vector_store = QdrantVectorStore(client=client, collection_name="paul_graham")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

"""
#### Query Index
"""
logger.info("#### Query Index")

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

display(Markdown(f"<b>{response}</b>"))

query_engine = index.as_query_engine()
response = query_engine.query(
    "What did the author do after his time at Viaweb?"
)

display(Markdown(f"<b>{response}</b>"))

"""
#### Build the VectorStoreIndex asynchronously
"""
logger.info("#### Build the VectorStoreIndex asynchronously")

# import nest_asyncio

# nest_asyncio.apply()

aclient = qdrant_client.AsyncQdrantClient(
    location=":memory:"
)

vector_store = QdrantVectorStore(
    collection_name="paul_graham",
    client=client,
    aclient=aclient,
    prefer_grpc=True,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    use_async=True,
)

"""
#### Async Query Index
"""
logger.info("#### Async Query Index")

query_engine = index.as_query_engine(use_async=True)
response = query_engine.query("What did the author do growing up?")
logger.success(format_json(response))

display(Markdown(f"<b>{response}</b>"))

query_engine = index.as_query_engine(use_async=True)
response = query_engine.query(
        "What did the author do after his time at Viaweb?"
    )
logger.success(format_json(response))

display(Markdown(f"<b>{response}</b>"))

"""
## Hybrid Search

You can enable hybrid search when creating an qdrant index. Here, we use Qdrant's BM25 capabilities to quickly create a sparse and dense index for hybrid retrieval.
"""
logger.info("## Hybrid Search")


client = QdrantClient(host="localhost", port=6333)
aclient = AsyncQdrantClient(host="localhost", port=6333)

vector_store = QdrantVectorStore(
    client=client,
    aclient=aclient,
    collection_name="paul_graham_hybrid",
    enable_hybrid=True,
    fastembed_sparse_model="Qdrant/bm25",
)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=StorageContext.from_defaults(vector_store=vector_store),
)

query_engine = index.as_query_engine(
    vector_store_query_mode="hybrid",
    sparse_top_k=2,
    similarity_top_k=2,
    hybrid_top_k=3,
)

response = query_engine.query("What did the author do growing up?")
display(Markdown(f"<b>{response}</b>"))

"""
## Saving and Loading

To restore an index, in most cases, you can just restore using the vector store object itself. The index is saved automatically by Qdrant.
"""
logger.info("## Saving and Loading")

loaded_index = VectorStoreIndex.from_vector_store(
    vector_store,
)

logger.info("\n\n[DONE]", bright=True)