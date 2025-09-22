from jet.logger import logger
from llama_index.core import StorageContext
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import VectorStoreIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
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
# Storing

Once you have data [loaded](/python/framework/module_guides/loading) and [indexed](/python/framework/module_guides/indexing), you will probably want to store it to avoid the time and cost of re-indexing it. By default, your indexed data is stored only in memory.

## Persisting to disk

The simplest way to store your indexed data is to use the built-in `.persist()` method of every Index, which writes all the data to disk at the location specified. This works for any type of index.
"""
logger.info("# Storing")

index.storage_context.persist(persist_dir="<persist_dir>")

"""
Here is an example of a Composable Graph:
"""
logger.info("Here is an example of a Composable Graph:")

graph.root_index.storage_context.persist(persist_dir="<persist_dir>")

"""
You can then avoid re-loading and re-indexing your data by loading the persisted index like this:
"""
logger.info("You can then avoid re-loading and re-indexing your data by loading the persisted index like this:")


storage_context = StorageContext.from_defaults(persist_dir="<persist_dir>")

index = load_index_from_storage(storage_context)

"""
<Aside type="tip">
Important: if you had initialized your index with a custom `transformations`, `embed_model`, etc., you will need to pass in the same options during `load_index_from_storage`, or have it set as the [global settings](/python/framework/module_guides/supporting_modules/settings).
</Aside>

## Using Vector Stores

As discussed in [indexing](/python/framework/module_guides/indexing), one of the most common types of Index is the VectorStoreIndex. The API calls to create the [embeddings](/python/framework/module_guides/indexing#what-is-an-embedding) in a VectorStoreIndex can be expensive in terms of time and money, so you will want to store them to avoid having to constantly re-index things.

LlamaIndex supports a [huge number of vector stores](/python/framework/module_guides/storing/vector_stores) which vary in architecture, complexity and cost. In this example we'll be using Chroma, an open-source vector store.

First you will need to install chroma:

# pip install chromadb

To use Chroma to store the embeddings from a VectorStoreIndex, you need to:

- initialize the Chroma client
- create a Collection to store your data in Chroma
- assign Chroma as the `vector_store` in a `StorageContext`
- initialize your VectorStoreIndex using that StorageContext

Here's what that looks like, with a sneak peek at actually querying the data:
"""
logger.info("## Using Vector Stores")


documents = SimpleDirectoryReader("./data").load_data()

db = chromadb.PersistentClient(path="./chroma_db")

chroma_collection = db.get_or_create_collection("quickstart")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

query_engine = index.as_query_engine()
response = query_engine.query("What is the meaning of life?")
logger.debug(response)

"""
If you've already created and stored your embeddings, you'll want to load them directly without loading your documents or creating a new VectorStoreIndex:
"""
logger.info("If you've already created and stored your embeddings, you'll want to load them directly without loading your documents or creating a new VectorStoreIndex:")


db = chromadb.PersistentClient(path="./chroma_db")

chroma_collection = db.get_or_create_collection("quickstart")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)

query_engine = index.as_query_engine()
response = query_engine.query("What is llama2?")
logger.debug(response)

"""
<Aside type="tip">
We have a [more thorough example of using Chroma](/python/examples/vector_stores/chromaindexdemo) if you want to go deeper on this store.
</Aside>

### You're ready to query!

Now you have loaded data, indexed it, and stored that index, you're ready to [query your data](/python/framework/module_guides/querying).

## Inserting Documents or Nodes

If you've already created an index, you can add new documents to your index using the `insert` method.
"""
logger.info("### You're ready to query!")


index = VectorStoreIndex([])
for doc in documents:
    index.insert(doc)

"""
See the [document management how-to](/python/framework/module_guides/indexing/document_management) for more details on managing documents and an example notebook.
"""
logger.info("See the [document management how-to](/python/framework/module_guides/indexing/document_management) for more details on managing documents and an example notebook.")

logger.info("\n\n[DONE]", bright=True)