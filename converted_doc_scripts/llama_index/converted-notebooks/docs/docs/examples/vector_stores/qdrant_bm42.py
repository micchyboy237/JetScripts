from fastembed import SparseTextEmbedding
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_parse import LlamaParse
import os
import qdrant_client
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Hybrid Search with Qdrant BM42

Qdrant recently released a new lightweight approach to sparse embeddings, [BM42](https://qdrant.tech/articles/bm42/).

In this notebook, we walk through how to use BM42 with llama-index, for effecient hybrid search.

## Setup

First, we need a few packages
- `llama-index`
- `llama-index-vector-stores-qdrant`
- `fastembed` or `fastembed-gpu`

`llama-index` will automatically run fastembed models on GPU if the provided libraries are installed. Check out their [full installation guide](https://qdrant.github.io/fastembed/examples/FastEmbed_GPU/).
"""
logger.info("# Hybrid Search with Qdrant BM42")

# %pip install llama-index llama-index-vector-stores-qdrant fastembed

"""
## (Optional) Test the fastembed package

To confirm the installation worked (and also to confirm GPU usage, if used), we can run the following code.

This will first download (and cache) the model locally, and then embed it.
"""
logger.info("## (Optional) Test the fastembed package")


model = SparseTextEmbedding(
    model_name="Qdrant/bm42-all-minilm-l6-v2-attentions",
)

embeddings = model.embed(["hello world", "goodbye world"])

indices, values = zip(
    *[
        (embedding.indices.tolist(), embedding.values.tolist())
        for embedding in embeddings
    ]
)

logger.debug(indices[0], values[0])

"""
## Construct our Hybrid Index

In llama-index, we can construct a hybrid index in just a few lines of code.

If you've tried hybrid in the past with splade, you will notice that this is much faster!

### Loading Data

Here, we use `llama-parse` to read in the Llama2 paper! Using the JSON result mode, we can get detailed data about each page, including layout and images. For now, we will use the page numbers and text.

You can get a free api key for `llama-parse` by visiting [https://cloud.llamaindex.ai](https://cloud.llamaindex.ai)
"""
logger.info("## Construct our Hybrid Index")

# !mkdir -p 'data/'
# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O f"{GENERATED_DIR}/llama2.pdf"

# import nest_asyncio

# nest_asyncio.apply()


parser = LlamaParse(result_type="text", api_key="llx-...")

json_data = parser.get_json_result(f"{GENERATED_DIR}/llama2.pdf")

documents = []
for document_json in json_data:
    for page in document_json["pages"]:
        documents.append(
            Document(text=page["text"], metadata={"page_number": page["page"]})
        )

"""
### Construct the Index /w Qdrant

With our nodes, we can construct our index with Qdrant and BM42!

In this case, Qdrant is being hosted in a docker container.

You can pull the latest:

```
docker pull qdrant/qdrant
```

And then to launch:

```
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```
"""
logger.info("### Construct the Index /w Qdrant")


client = qdrant_client.QdrantClient("http://localhost:6333")
aclient = qdrant_client.AsyncQdrantClient("http://localhost:6333")

if client.collection_exists("llama2_bm42"):
    client.delete_collection("llama2_bm42")

vector_store = QdrantVectorStore(
    collection_name="llama2_bm42",
    client=client,
    aclient=aclient,
    fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
)


storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=MLXEmbedding(
        model_name="mxbai-embed-large", api_key="sk-proj-..."
    ),
    storage_context=storage_context,
)

"""
As we can see, both the dense and sparse embeddings were generated super quickly!

Even though the sparse model is running locally on CPU, its very small and fast.

## Test out the Index

Using the powers of sparse embeddings, we can query for some very specific facts, and get the correct data.
"""
logger.info("## Test out the Index")


chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    llm=MLX(model="qwen3-1.7b-4bit", api_key="sk-proj-..."),
)

response = chat_engine.chat("What training hardware was used for Llama2?")
logger.debug(str(response))

response = chat_engine.chat("What is the main idea of Llama2?")
logger.debug(str(response))

response = chat_engine.chat("What was Llama2 evaluated and compared against?")
logger.debug(str(response))

"""
## Loading from existing store

With your vector index created, we can easily connect back to it!
"""
logger.info("## Loading from existing store")


client = qdrant_client.QdrantClient("http://localhost:6333")
aclient = qdrant_client.AsyncQdrantClient("http://localhost:6333")

if client.collection_exists("llama2_bm42"):
    client.delete_collection("llama2_bm42")

vector_store = QdrantVectorStore(
    collection_name="llama2_bm42",
    client=client,
    aclient=aclient,
    fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
)

loaded_index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=MLXEmbedding(
        model="mxbai-embed-large", api_key="sk-proj-..."
    ),
)

logger.info("\n\n[DONE]", bright=True)