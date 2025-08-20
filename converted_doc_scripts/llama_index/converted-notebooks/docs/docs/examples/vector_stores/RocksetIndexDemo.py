from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.vector_stores.types import NodeWithEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.rocksetdb import RocksetVectorStore
from rockset import Regions
from rockset import RocksetClient
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/RocksetIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Rockset Vector Store

As a real-time search and analytics database, Rockset uses indexing to deliver scalable and performant personalization, product search, semantic search, chatbot applications, and more.
Since Rockset is purpose-built for real-time, you can build these responsive applications on constantly updating, streaming data. 
By integrating Rockset with LlamaIndex, you can easily use LLMs on your own real-time data for production-ready vector search applications.

We'll walk through a demonstration of how to use Rockset as a vector store in LlamaIndex. 

## Tutorial
In this example, we'll use MLX's `text-embedding-ada-002` model to generate embeddings and Rockset as vector store to store embeddings.
We'll ingest text from a file and ask questions about the content.

### Setting Up Your Environment
1. Create a [collection](https://rockset.com/docs/collections) from the Rockset console with the [Write API](https://rockset.com/docs/write-api/) as your source.
Name your collection `llamaindex_demo`. Configure the following [ingest transformation](https://rockset.com/docs/ingest-transformation) 
with [`VECTOR_ENFORCE`](https://rockset.com/docs/vector-functions) to define your embeddings field and take advantage of performance and storage optimizations:
```sql
SELECT 
    _input.* EXCEPT(_meta), 
    VECTOR_ENFORCE(
        _input.embedding,
        1536,
        'float'
    ) as embedding
FROM _input
```

2. Create an [API key](https://rockset.com/docs/iam) from the Rockset console and set the `ROCKSET_API_KEY` environment variable.
Find your API server [here](http://rockset.com/docs/rest-api#introduction) and set the `ROCKSET_API_SERVER` environment variable. 
# Set the `OPENAI_API_KEY` environment variable.

3. Install the dependencies.
```shell
pip3 install llama_index rockset 
```

4. LlamaIndex allows you to ingest data from a variety of sources. 
For this example, we'll read from a text file named `constitution.txt`, which is a transcript of the American Constitution, found [here](https://www.archives.gov/founding-docs/constitution-transcript). 

### Data ingestion 
Use LlamaIndex's `SimpleDirectoryReader` class to convert the text file to a list of `Document` objects.
"""
logger.info("# Rockset Vector Store")

# %pip install llama-index-llms-ollama
# %pip install llama-index-vector-stores-rocksetdb


docs = SimpleDirectoryReader(
    input_files=["{path to}/consitution.txt"]
).load_data()

"""
Instantiate the LLM and service context.
"""
logger.info("Instantiate the LLM and service context.")


Settings.llm = MLX(temperature=0.8, model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

"""
Instantiate the vector store and storage context.
"""
logger.info("Instantiate the vector store and storage context.")


vector_store = RocksetVectorStore(collection="llamaindex_demo")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

"""
Add documents to the `llamaindex_demo` collection and create an index.
"""
logger.info("Add documents to the `llamaindex_demo` collection and create an index.")


index = VectorStoreIndex.from_documents(
    docs,
    storage_context=storage_context,
)

"""
### Querying
Ask a question about your document and generate a response.
"""
logger.info("### Querying")

response = index.as_query_engine().query("What is the duty of the president?")

logger.debug(str(response))

"""
Run the program.
```text
$ python3 main.py
The duty of the president is to faithfully execute the Office of President of the United States, preserve, protect and defend the Constitution of the United States, serve as the Commander in Chief of the Army and Navy, grant reprieves and pardons for offenses against the United States (except in cases of impeachment), make treaties and appoint ambassadors and other public ministers, take care that the laws be faithfully executed, and commission all the officers of the United States.
```

## Metadata Filtering
Metadata filtering allows you to retrieve relevant documents that match specific filters.

1. Add nodes to your vector store and create an index.
"""
logger.info("## Metadata Filtering")


nodes = [
    NodeWithEmbedding(
        node=TextNode(
            text="Apples are blue",
            metadata={"type": "fruit"},
        ),
        embedding=[],
    )
]
index = VectorStoreIndex(
    nodes,
    storage_context=StorageContext.from_defaults(
        vector_store=RocksetVectorStore(collection="llamaindex_demo")
    ),
)

"""
2. Define metadata filters.
"""
logger.info("2. Define metadata filters.")


filters = MetadataFilters(
    filters=[ExactMatchFilter(key="type", value="fruit")]
)

"""
3. Retrieve relevant documents that satisfy the filters.
"""
logger.info("3. Retrieve relevant documents that satisfy the filters.")

retriever = index.as_retriever(filters=filters)
retriever.retrieve("What colors are apples?")

"""
## Creating an Index from an Existing Collection
You can create indices with data from existing collections.
"""
logger.info("## Creating an Index from an Existing Collection")


vector_store = RocksetVectorStore(collection="llamaindex_demo")

index = VectorStoreIndex.from_vector_store(vector_store)

"""
## Creating an Index from a New Collection
You can also create a new Rockset collection to use as a vector store.
"""
logger.info("## Creating an Index from a New Collection")


vector_store = RocksetVectorStore.with_new_collection(
    collection="llamaindex_demo",  # name of new collection
    dimensions=1536,  # specifies length of vectors in ingest tranformation (optional)
)

index = VectorStoreIndex(
    nodes,
    storage_context=StorageContext.from_defaults(vector_store=vector_store),
)

"""
## Configuration
* **collection**: Name of the collection to query (required).

```python
RocksetVectorStore(collection="my_collection")
```

* **workspace**: Name of the workspace containing the collection. Defaults to `"commons"`.
```python
RocksetVectorStore(worksapce="my_workspace")
```

* **api_key**: The API key to use to authenticate Rockset requests. Ignored if `client` is passed in. Defaults to the `ROCKSET_API_KEY` environment variable.
```python
RocksetVectorStore(api_key="<my key>")
```

* **api_server**: The API server to use for Rockset requests. Ignored if `client` is passed in. Defaults to the `ROCKSET_API_KEY` environment variable or `"https://api.use1a1.rockset.com"` if the `ROCKSET_API_SERVER` is not set.
```python
RocksetVectorStore(api_server=Regions.euc1a1)
```

* **client**: Rockset client object to use to execute Rockset requests. If not specified, a client object is internally constructed with the `api_key` parameter (or `ROCKSET_API_SERVER` environment variable) and the `api_server` parameter (or `ROCKSET_API_SERVER` environment variable).
```python
RocksetVectorStore(client=RocksetClient(api_key="<my key>"))
```

* **embedding_col**: The name of the database field containing embeddings. Defaults to `"embedding"`.
```python
RocksetVectorStore(embedding_col="my_embedding")
```

* **metadata_col**: The name of the database field containing node data. Defaults to `"metadata"`.
```python
RocksetVectorStore(metadata_col="node")
```

* **distance_func**: The metric to measure vector relationship. Defaults to cosine similarity.
```python
RocksetVectorStore(distance_func=RocksetVectorStore.DistanceFunc.DOT_PRODUCT)
```
"""
logger.info("## Configuration")

logger.info("\n\n[DONE]", bright=True)