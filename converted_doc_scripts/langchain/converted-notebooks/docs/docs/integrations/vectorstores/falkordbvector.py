from jet.models.config import MODELS_CACHE_DIR
from jet.logger import logger
from langchain_community.vectorstores.falkordb_vector import FalkorDBVector
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
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
# FalkorDBVectorStore
<a href="https://docs.falkordb.com/" target="_blank">FalkorDB</a> is an open-source graph database with integrated support for vector similarity search

it supports:
- approximate nearest neighbor search
- Euclidean similarity & Cosine Similarity
- Hybrid search combining vector and keyword searches

This notebook shows how to use the FalkorDB vector index (`FalkorDB`)

See the <a href="https://docs.falkordb.com/" target="_blank">installation instruction</a>

## Setup
"""
logger.info("# FalkorDBVectorStore")

# %pip install --upgrade  falkordb
# %pip install --upgrade  tiktoken
# %pip install --upgrade  langchain langchain_huggingface

"""
### Credentials
We want to use `HuggingFace` so we have to get the HuggingFace API Key
"""
logger.info("### Credentials")

# import getpass

if "HUGGINGFACE_API_KEY" not in os.environ:
#     os.environ["HUGGINGFACE_API_KEY"] = getpass.getpass("HuggingFace API Key:")

"""
If you want to get automated tracing of your model calls you can also set your LangSmith API key by uncommenting below:
"""
logger.info("If you want to get automated tracing of your model calls you can also set your LangSmith API key by uncommenting below:")



"""
## Initialization
"""
logger.info("## Initialization")


"""
You can use FalkorDBVector locally with docker. See <a href="https://docs.falkordb.com/" target="_blank">installation instruction</a>
"""
logger.info("You can use FalkorDBVector locally with docker. See <a href="https://docs.falkordb.com/" target="_blank">installation instruction</a>")

host = "localhost"
port = 6379

"""
Or you can use FalkorDBVector with <a href="https://app.falkordb.cloud">FalkorDB Cloud</a>
"""
logger.info("Or you can use FalkorDBVector with <a href="https://app.falkordb.cloud">FalkorDB Cloud</a>")



vector_store = FalkorDBVector(host=host, port=port, embedding=HuggingFaceEmbeddings())

"""
## Manage vector store

### Add items to vector store
"""
logger.info("## Manage vector store")


document_1 = Document(page_content="foo", metadata={"source": "https://example.com"})

document_2 = Document(page_content="bar", metadata={"source": "https://example.com"})

document_3 = Document(page_content="baz", metadata={"source": "https://example.com"})

documents = [document_1, document_2, document_3]

vector_store.add_documents(documents=documents, ids=["1", "2", "3"])

"""
### Update items in vector store
"""
logger.info("### Update items in vector store")

updated_document = Document(
    page_content="qux", metadata={"source": "https://another-example.com"}
)

vector_store.update_documents(document_id="1", document=updated_document)

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

results = vector_store.similarity_search_with_score(query="bar")
for doc, score in results:
    logger.debug(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

"""
### Query by turning into retriever
You can also transform the vector store into a retriever for easier usage in your chains.
"""
logger.info("### Query by turning into retriever")

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
retriever.invoke("thud")

"""
## Usage for retrieval-augmented generation
For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:
- <a href="https://python.langchain.com/v0.2/docs/tutorials/#working-with-external-knowledge" target="_blank">Tutorials: working with external knowledge</a>
- <a href="https://python.langchain.com/v0.2/docs/how_to/#qa-with-rag" target="_blank">How-to: Question and answer with RAG</a>
- <a href="Retrieval conceptual docs" target="_blank">Retrieval conceptual docs</a>

## API reference
For detailed documentation of all `FalkorDBVector` features and configurations head to the API reference: https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.falkordb_vector.FalkorDBVector.html
"""
logger.info("## Usage for retrieval-augmented generation")

logger.info("\n\n[DONE]", bright=True)