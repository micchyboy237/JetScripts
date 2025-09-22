from dotenv import load_dotenv
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_core.documents import Document
from langchain_zeusdb import ZeusDBVectorStore
from zeusdb import VectorDatabase
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
# ⚡ ZeusDB Vector Store

ZeusDB is a high-performance, Rust-powered vector database with enterprise features like quantization, persistence and logging.

This notebook covers how to get started with the ZeusDB Vector Store to efficiently use ZeusDB with LangChain.

---

## Setup

Install the ZeusDB LangChain integration package from PyPi:
"""
logger.info("# ⚡ ZeusDB Vector Store")

pip install -qU langchain-zeusdb

"""
*Setup in Jupyter Notebooks*

> 💡 Tip: If you’re working inside Jupyter or Google Colab, use the %pip magic command so the package is installed into the active kernel:
"""

# %pip install -qU langchain-zeusdb

"""
---

## Getting Started

This example uses OllamaEmbeddings, which requires an Ollama API key – [Get your Ollama API key here](https://platform.ollama.com/api-keys)

If you prefer, you can also use this package with any other embedding provider (Hugging Face, Cohere, custom functions, etc.).

Install the LangChain Ollama integration package from PyPi:
"""
logger.info("## Getting Started")

pip install -qU langchain-ollama

"""
#### Please choose an option below for your Ollama key integration

*Option 1: 🔑 Enter your API key each time*

# Use getpass in Jupyter to securely input your key for the current session:
"""
logger.info("#### Please choose an option below for your Ollama key integration")

# import getpass

# os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")

"""
*Option 2: 🗂️ Use a .env file*

Keep your key in a local .env file and load it automatically with python-dotenv
"""
logger.info("Keep your key in a local .env file and load it automatically with python-dotenv")


# load_dotenv()  # reads .env and sets OPENAI_API_KEY

"""
🎉🎉 That's it! You are good to go.

---

## Initialization
"""
logger.info("## Initialization")


embeddings = OllamaEmbeddings(model="nomic-embed-text")

vdb = VectorDatabase()
index = vdb.create(index_type="hnsw", dim=1536, space="cosine")

vector_store = ZeusDBVectorStore(zeusdb_index=index, embedding=embeddings)

"""
---

## Manage vector store

### 2.1 Add items to vector store
"""
logger.info("## Manage vector store")


document_1 = Document(
    page_content="ZeusDB is a high-performance vector database",
    metadata={"source": "https://docs.zeusdb.com"},
)

document_2 = Document(
    page_content="Product Quantization reduces memory usage significantly",
    metadata={"source": "https://docs.zeusdb.com"},
)

document_3 = Document(
    page_content="ZeusDB integrates seamlessly with LangChain",
    metadata={"source": "https://docs.zeusdb.com"},
)

documents = [document_1, document_2, document_3]

vector_store.add_documents(documents=documents, ids=["1", "2", "3"])

"""
### 2.2 Update items in vector store
"""
logger.info("### 2.2 Update items in vector store")

updated_document = Document(
    page_content="ZeusDB now supports advanced Product Quantization with 4x-256x compression",
    metadata={"source": "https://docs.zeusdb.com", "updated": True},
)

vector_store.add_documents([updated_document], ids=["1"])

"""
### 2.3 Delete items from vector store
"""
logger.info("### 2.3 Delete items from vector store")

vector_store.delete(ids=["3"])

"""
---

## Query vector store

### 3.1 Query directly

Performing a simple similarity search:
"""
logger.info("## Query vector store")

results = vector_store.similarity_search(query="high performance database", k=2)

for doc in results:
    logger.debug(f"* {doc.page_content} [{doc.metadata}]")

"""
If you want to execute a similarity search and receive the corresponding scores:
"""
logger.info("If you want to execute a similarity search and receive the corresponding scores:")

results = vector_store.similarity_search_with_score(query="memory optimization", k=2)

for doc, score in results:
    logger.debug(f"* [SIM={score:.3f}] {doc.page_content} [{doc.metadata}]")

"""
### 3.2 Query by turning into retriever

You can also transform the vector store into a retriever for easier usage in your chains:
"""
logger.info("### 3.2 Query by turning into retriever")

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})

retriever.invoke("vector database features")

"""
---

## ZeusDB-Specific Features

### 4.1 Memory-Efficient Setup with Product Quantization

For large datasets, use Product Quantization to reduce memory usage:
"""
logger.info("## ZeusDB-Specific Features")

quantization_config = {"type": "pq", "subvectors": 8, "bits": 8, "training_size": 10000}

vdb_quantized = VectorDatabase()
quantized_index = vdb_quantized.create(
    index_type="hnsw", dim=1536, quantization_config=quantization_config
)

quantized_vector_store = ZeusDBVectorStore(
    zeusdb_index=quantized_index, embedding=embeddings
)

logger.debug(f"Created quantized store: {quantized_index.info()}")

"""
### 4.2 Persistence

Save and load your vector store to disk:

How to Save your vector store
"""
logger.info("### 4.2 Persistence")

vector_store.save_index("my_zeusdb_index.zdb")

"""
How to Load your vector store
"""
logger.info("How to Load your vector store")

loaded_store = ZeusDBVectorStore.load_index(
    path="my_zeusdb_index.zdb", embedding=embeddings
)

logger.debug(f"Loaded store with {loaded_store.get_vector_count()} vectors")

"""
---

## Usage for retrieval-augmented generation

For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:

- [How-to: Question and answer with RAG](https://python.langchain.com/docs/how_to/#qa-with-rag)
- [Retrieval conceptual docs](https://python.langchain.com/docs/concepts/retrieval/)

---

## API reference

For detailed documentation of all ZeusDBVectorStore features and configurations head to the Doc reference: https://docs.zeusdb.com/en/latest/vector_database/integrations/langchain.html
"""
logger.info("## Usage for retrieval-augmented generation")

logger.info("\n\n[DONE]", bright=True)