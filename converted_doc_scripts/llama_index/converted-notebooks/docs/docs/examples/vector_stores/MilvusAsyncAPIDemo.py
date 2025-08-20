import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
import asyncio
import os
import random
import shutil
import time


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/MilvusAsyncAPIDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Milvus Vector Store with Async API

This tutorial demonstrates how to use [LlamaIndex](https://www.llamaindex.ai/) with [Milvus](https://milvus.io/) to build asynchronous document processing pipeline for RAG. LlamaIndex provides a way to process documents and store in vector db like Milvus. By leveraging the async API of LlamaIndex and Milvus Python client library, we can increase the throughput of the pipeline to efficiently process and index large volumes of data.

 
In this tutorial, we will first introduce the use of asynchronous methods to build a RAG with LlamaIndex and Milvus from a high level, and then introduce the use of low level methods and the performance comparison of synchronous and asynchronous.

## Before you begin

Code snippets on this page require pymilvus and llamaindex dependencies. You can install them using the following commands:
"""
logger.info("# Milvus Vector Store with Async API")

# ! pip install -U pymilvus llama-index-vector-stores-milvus llama-index nest-asyncio

"""
> If you are using Google Colab, to enable dependencies just installed, you may need to **restart the runtime** (click on the "Runtime" menu at the top of the screen, and select "Restart session" from the dropdown menu).

# We will use the models from MLX. You should prepare the [api key](https://platform.openai.com/docs/quickstart) `OPENAI_API_KEY` as an environment variable.
"""
# logger.info("We will use the models from MLX. You should prepare the [api key](https://platform.openai.com/docs/quickstart) `OPENAI_API_KEY` as an environment variable.")


# os.environ["OPENAI_API_KEY"] = "sk-***********"

"""
If you are using Jupyter Notebook, you need to run this line of code before running the asynchronous code.
"""
logger.info("If you are using Jupyter Notebook, you need to run this line of code before running the asynchronous code.")

# import nest_asyncio

# nest_asyncio.apply()

"""
### Prepare data

You can download sample data with the following commands:
"""
logger.info("### Prepare data")

# ! mkdir -p 'data/'
# ! wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham_essay.txt'
# ! wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/uber_2021.pdf'

"""
## Build RAG with Asynchronous Processing
This section show how to build a RAG system that can process docs in asynchronous manner.

Import the necessary libraries and define Milvus URI and the dimension of the embedding.
"""
logger.info("## Build RAG with Asynchronous Processing")



URI = "http://localhost:19530"
DIM = 768

"""
> - If you have large scale of data, you can set up a performant Milvus server on [docker or kubernetes](https://milvus.io/docs/quickstart.md). In this setup, please use the server uri, e.g.`http://localhost:19530`, as your `uri`.
> - If you want to use [Zilliz Cloud](https://zilliz.com/cloud), the fully managed cloud service for Milvus, adjust the `uri` and `token`, which correspond to the [Public Endpoint and Api key](https://docs.zilliz.com/docs/on-zilliz-cloud-console#free-cluster-details) in Zilliz Cloud.
> - In the case of complex systems (such as network communication), asynchronous processing can bring performance improvement compared to synchronization. So we think Milvus-Lite is not suitable for using asynchronous interfaces because the scenarios used are not suitable.

Define an initialization function that we can use again to rebuild the Milvus collection.
"""
logger.info("Define an initialization function that we can use again to rebuild the Milvus collection.")

def init_vector_store():
    return MilvusVectorStore(
        uri=URI,
        dim=DIM,
        collection_name="test_collection",
        embedding_field="embedding",
        id_field="id",
        similarity_metric="COSINE",
        consistency_level="Strong",
        overwrite=True,  # To overwrite the collection if it already exists
    )


vector_store = init_vector_store()

"""
Use SimpleDirectoryReader to wrap a LlamaIndex document object from the file `paul_graham_essay.txt`.
"""
logger.info("Use SimpleDirectoryReader to wrap a LlamaIndex document object from the file `paul_graham_essay.txt`.")


documents = SimpleDirectoryReader(
    input_files=["/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data_essay.txt"]
).load_data()

logger.debug("Document ID:", documents[0].doc_id)

"""
Instantiate a Hugging Face embedding model locally. Using a local model avoids the risk of reaching API rate limits during asynchronous data insertion, as concurrent API requests can quickly add up and use up your budget in public API. However, if you have a high rate limit, you may opt to use a remote model service instead.
"""
logger.info("Instantiate a Hugging Face embedding model locally. Using a local model avoids the risk of reaching API rate limits during asynchronous data insertion, as concurrent API requests can quickly add up and use up your budget in public API. However, if you have a high rate limit, you may opt to use a remote model service instead.")



embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

"""
Create an index and insert the document.

We set the `use_async` to `True` to enable async insert mode.
"""
logger.info("Create an index and insert the document.")


storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
    use_async=True,
)

"""
Initialize the LLM.
"""
logger.info("Initialize the LLM.")


llm = MLX(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

"""
When building the query engine, you can also set the `use_async` parameter to `True` to enable asynchronous search.
"""
logger.info("When building the query engine, you can also set the `use_async` parameter to `True` to enable asynchronous search.")

query_engine = index.as_query_engine(use_async=True, llm=llm)
async def run_async_code_3dec1504():
    async def run_async_code_48b10a9f():
        response = query_engine.query("What did the author learn?")
        return response
    response = asyncio.run(run_async_code_48b10a9f())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_3dec1504())
logger.success(format_json(response))

logger.debug(response)

"""
## Explore the Async API

In this section, we'll introduce lower level API usage and compare the performance of synchronous and asynchronous runs.

### Async add
Re-initialize the vector store.
"""
logger.info("## Explore the Async API")

vector_store = init_vector_store()

"""
Let's define a node producing function, which will be used to generate large number of test nodes for the index.
"""
logger.info("Let's define a node producing function, which will be used to generate large number of test nodes for the index.")

def random_id():
    random_num_str = ""
    for _ in range(16):
        random_digit = str(random.randint(0, 9))
        random_num_str += random_digit
    return random_num_str


def produce_nodes(num_adding):
    node_list = []
    for i in range(num_adding):
        node = TextNode(
            id_=random_id(),
            text=f"n{i}_text",
            embedding=[0.5] * (DIM - 1) + [random.random()],
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id=f"n{i+1}")
            },
        )
        node_list.append(node)
    return node_list

"""
Define a aync function to add documents to the vector store. We use the `async_add()` function in Milvus vector store instance.
"""
logger.info("Define a aync function to add documents to the vector store. We use the `async_add()` function in Milvus vector store instance.")

async def async_add(num_adding):
    node_list = produce_nodes(num_adding)
    start_time = time.time()
    tasks = []
    for i in range(num_adding):
        sub_nodes = node_list[i]
        task = vector_store.async_add([sub_nodes])  # use async_add()
        tasks.append(task)
    async def run_async_code_9aa35d2c():
        async def run_async_code_20dc84be():
            results = await asyncio.gather(*tasks)
            return results
        results = asyncio.run(run_async_code_20dc84be())
        logger.success(format_json(results))
        return results
    results = asyncio.run(run_async_code_9aa35d2c())
    logger.success(format_json(results))
    end_time = time.time()
    return end_time - start_time

add_counts = [10, 100, 1000]

"""
Get the event loop.
"""
logger.info("Get the event loop.")

loop = asyncio.get_event_loop()

"""
Asynchronously add documents to the vector store.
"""
logger.info("Asynchronously add documents to the vector store.")

for count in add_counts:

    async def measure_async_add():
        async def run_async_code_bc508403():
            async def run_async_code_65862829():
                async_time = await async_add(count)
                return async_time
            async_time = asyncio.run(run_async_code_65862829())
            logger.success(format_json(async_time))
            return async_time
        async_time = asyncio.run(run_async_code_bc508403())
        logger.success(format_json(async_time))
        logger.debug(f"Async add for {count} took {async_time:.2f} seconds")
        return async_time

    loop.run_until_complete(measure_async_add())

vector_store = init_vector_store()

"""
#### Compare with synchronous add
Define a sync add function. Then measure the running time under the same condition.
"""
logger.info("#### Compare with synchronous add")

def sync_add(num_adding):
    node_list = produce_nodes(num_adding)
    start_time = time.time()
    for node in node_list:
        result = vector_store.add([node])
    end_time = time.time()
    return end_time - start_time

for count in add_counts:
    sync_time = sync_add(count)
    logger.debug(f"Sync add for {count} took {sync_time:.2f} seconds")

"""
The result shows that the sync adding process is much slower than the async one.

### Async search

Re-initialize the vector store and add some documents before running the search.
"""
logger.info("### Async search")

vector_store = init_vector_store()
node_list = produce_nodes(num_adding=1000)
inserted_ids = vector_store.add(node_list)

"""
Define an async search function. We use the `aquery()` function in Milvus vector store instance.
"""
logger.info("Define an async search function. We use the `aquery()` function in Milvus vector store instance.")

async def async_search(num_queries):
    start_time = time.time()
    tasks = []
    for _ in range(num_queries):
        query = VectorStoreQuery(
            query_embedding=[0.5] * (DIM - 1) + [0.6], similarity_top_k=3
        )
        task = vector_store.aquery(query=query)  # use aquery()
        tasks.append(task)
    async def run_async_code_9aa35d2c():
        async def run_async_code_20dc84be():
            results = await asyncio.gather(*tasks)
            return results
        results = asyncio.run(run_async_code_20dc84be())
        logger.success(format_json(results))
        return results
    results = asyncio.run(run_async_code_9aa35d2c())
    logger.success(format_json(results))
    end_time = time.time()
    return end_time - start_time

query_counts = [10, 100, 1000]

"""
Asynchronously search from Milvus store.
"""
logger.info("Asynchronously search from Milvus store.")

for count in query_counts:

    async def measure_async_search():
        async def run_async_code_ae25e6a0():
            async def run_async_code_2bdc493c():
                async_time = await async_search(count)
                return async_time
            async_time = asyncio.run(run_async_code_2bdc493c())
            logger.success(format_json(async_time))
            return async_time
        async_time = asyncio.run(run_async_code_ae25e6a0())
        logger.success(format_json(async_time))
        logger.debug(
            f"Async search for {count} queries took {async_time:.2f} seconds"
        )
        return async_time

    loop.run_until_complete(measure_async_search())

"""
#### Compare with synchronous search
Define a sync search function. Then measure the running time under the same condition.
"""
logger.info("#### Compare with synchronous search")

def sync_search(num_queries):
    start_time = time.time()
    for _ in range(num_queries):
        query = VectorStoreQuery(
            query_embedding=[0.5] * (DIM - 1) + [0.6], similarity_top_k=3
        )
        result = vector_store.query(query=query)
    end_time = time.time()
    return end_time - start_time

for count in query_counts:
    sync_time = sync_search(count)
    logger.debug(f"Sync search for {count} queries took {sync_time:.2f} seconds")

"""
The result shows that the sync search process is much slower than the async one.
"""
logger.info("The result shows that the sync search process is much slower than the async one.")

logger.info("\n\n[DONE]", bright=True)