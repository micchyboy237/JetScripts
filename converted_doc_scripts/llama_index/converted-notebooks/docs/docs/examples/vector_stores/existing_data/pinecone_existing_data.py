from jet.models.config import MODELS_CACHE_DIR
from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex
from llama_index.core.response.pprint_utils import pprint_source_node
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
import os
import pinecone
import shutil
import uuid


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/existing_data/pinecone_existing_data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Guide: Using Vector Store Index with Existing Pinecone Vector Store

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Guide: Using Vector Store Index with Existing Pinecone Vector Store")

# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-vector-stores-pinecone

# !pip install llama-index


api_key = os.environ["PINECONE_API_KEY"]
pinecone.init(api_key=api_key, environment="eu-west1-gcp")

"""
## Prepare Sample "Existing" Pinecone Vector Store

### Create index
"""
logger.info("## Prepare Sample "Existing" Pinecone Vector Store")

indexes = pinecone.list_indexes()
logger.debug(indexes)

if "quickstart-index" not in indexes:
    pinecone.create_index(
        "quickstart-index", dimension=1536, metric="euclidean", pod_type="p1"
    )

pinecone_index = pinecone.Index("quickstart-index")

pinecone_index.delete(deleteAll="true")

"""
### Define sample data
We create 4 sample books
"""
logger.info("### Define sample data")

books = [
    {
        "title": "To Kill a Mockingbird",
        "author": "Harper Lee",
        "content": (
            "To Kill a Mockingbird is a novel by Harper Lee published in"
            " 1960..."
        ),
        "year": 1960,
    },
    {
        "title": "1984",
        "author": "George Orwell",
        "content": (
            "1984 is a dystopian novel by George Orwell published in 1949..."
        ),
        "year": 1949,
    },
    {
        "title": "The Great Gatsby",
        "author": "F. Scott Fitzgerald",
        "content": (
            "The Great Gatsby is a novel by F. Scott Fitzgerald published in"
            " 1925..."
        ),
        "year": 1925,
    },
    {
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "content": (
            "Pride and Prejudice is a novel by Jane Austen published in"
            " 1813..."
        ),
        "year": 1813,
    },
]

"""
### Add data
We add the sample books to our Weaviate "Book" class (with embedding of content field
"""
logger.info("### Add data")


embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)

entries = []
for book in books:
    vector = embed_model.get_text_embedding(book["content"])
    entries.append(
        {"id": str(uuid.uuid4()), "values": vector, "metadata": book}
    )
pinecone_index.upsert(entries)

"""
## Query Against "Existing" Pinecone Vector Store
"""
logger.info("## Query Against "Existing" Pinecone Vector Store")


"""
You must properly select a class property as the "text" field.
"""
logger.info("You must properly select a class property as the "text" field.")

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index, text_key="content"
)

retriever = VectorStoreIndex.from_vector_store(vector_store).as_retriever(
    similarity_top_k=1
)

nodes = retriever.retrieve("What is that book about a bird again?")

"""
Let's inspect the retrieved node. We can see that the book data is loaded as LlamaIndex `Node` objects, with the "content" field as the main text.
"""
logger.info("Let's inspect the retrieved node. We can see that the book data is loaded as LlamaIndex `Node` objects, with the "content" field as the main text.")

pprint_source_node(nodes[0])

"""
The remaining fields should be loaded as metadata (in `metadata`)
"""
logger.info("The remaining fields should be loaded as metadata (in `metadata`)")

nodes[0].node.metadata

logger.info("\n\n[DONE]", bright=True)