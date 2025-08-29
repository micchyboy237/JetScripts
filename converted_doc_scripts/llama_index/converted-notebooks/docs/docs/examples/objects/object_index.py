from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core import SummaryIndex
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SimpleObjectNodeMapping
from llama_index.core.objects import SimpleToolNodeMapping
from llama_index.core.schema import TextNode
from llama_index.core.tools import FunctionTool
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/objects/object_index.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# The `ObjectIndex` Class

The `ObjectIndex` class is one that allows for the indexing of arbitrary Python objects. As such, it is quite flexible and applicable to a wide-range of use cases. As examples:
- [Use an `ObjectIndex` to index Tool objects to then be used by an agent.](https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent_retrieval.html#building-an-object-index)
- [Use an `ObjectIndex` to index a SQLTableSchema objects](https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo.html#part-2-query-time-retrieval-of-tables-for-text-to-sql)

To construct an `ObjectIndex`, we require an index as well as another abstraction, namely `ObjectNodeMapping`. This mapping, as its name suggests, provides the means to go between node and the associated object, and vice versa. Alternatively, there exists a `from_objects()` class method, that can conveniently construct an `ObjectIndex` from a set of objects.

In this notebook, we'll quickly cover how you can build an `ObjectIndex` using a `SimpleObjectNodeMapping`.
"""
logger.info("# The `ObjectIndex` Class")


Settings.embed_model = "local"


obj1 = {"input": "Hey, how's it going"}
obj2 = ["a", "b", "c", "d"]
obj3 = "llamaindex is an awesome library!"
arbitrary_objects = [obj1, obj2, obj3]

obj_node_mapping = SimpleObjectNodeMapping.from_objects(arbitrary_objects)
nodes = obj_node_mapping.to_nodes(arbitrary_objects)

object_index = ObjectIndex(
    index=VectorStoreIndex(nodes=nodes),
    object_node_mapping=obj_node_mapping,
)

object_index = ObjectIndex.from_objects(
    arbitrary_objects, index_cls=VectorStoreIndex
)

"""
### As a retriever
With the `object_index` in hand, we can use it as a retriever, to retrieve against the index objects.
"""
logger.info("### As a retriever")

object_retriever = object_index.as_retriever(similarity_top_k=1)
object_retriever.retrieve("llamaindex")

"""
We can also add node-postprocessors to an object index retriever, for easy convience to things like rerankers and more.
"""
logger.info("We can also add node-postprocessors to an object index retriever, for easy convience to things like rerankers and more.")

# %pip install llama-index-postprocessor-colbert-rerank


retriever = object_index.as_retriever(
    similarity_top_k=2, node_postprocessors=[ColbertRerank(top_n=1)]
)
retriever.retrieve("a random list object")

"""
## Using a Storage Integration (i.e. Chroma)

The object index supports integrations with any existing storage backend in LlamaIndex.

The following section walks through how to set that up using `Chroma` as an example.
"""
logger.info("## Using a Storage Integration (i.e. Chroma)")

# %pip install llama-index-vector-stores-chroma


db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart2")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

object_index = ObjectIndex.from_objects(
    arbitrary_objects,
    index_cls=VectorStoreIndex,
    storage_context=storage_context,
)

object_retriever = object_index.as_retriever(similarity_top_k=1)
object_retriever.retrieve("llamaindex")

"""
Now, lets "reload" the index
"""
logger.info("Now, lets "reload" the index")

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

object_index = ObjectIndex.from_objects_and_index(arbitrary_objects, index)

object_retriever = object_index.as_retriever(similarity_top_k=1)
object_retriever.retrieve("llamaindex")

"""
Note that when we reload the index, we still have to pass the objects, since those are not saved in the actual index/vector db.

## [Advanced] Customizing the Mapping

For specialized cases where you want full control over how objects are mapped to nodes, you can also provide a `to_node_fn()` and `from_node_fn()` hook.

This is useful for when you are converting specialized objects, or want to dynamically create objects at runtime rather than keeping them in memory.

A small example is shown below.
"""
logger.info("## [Advanced] Customizing the Mapping")


my_objects = {
    str(hash(str(obj))): obj for i, obj in enumerate(arbitrary_objects)
}


def from_node_fn(node):
    return my_objects[node.id]


def to_node_fn(obj):
    return TextNode(id=str(hash(str(obj))), text=str(obj))


object_index = ObjectIndex.from_objects(
    arbitrary_objects,
    index_cls=VectorStoreIndex,
    from_node_fn=from_node_fn,
    to_node_fn=to_node_fn,
)

object_retriever = object_index.as_retriever(similarity_top_k=1)

object_retriever.retrieve("llamaindex")

"""
## Persisting `ObjectIndex` to Disk with Objects

When it comes to persisting the `ObjectIndex`, we have to handle both the index as well as the object-node mapping. Persisting the index is straightforward and can be handled by usual means (e.g., see this [guide](https://docs.llamaindex.ai/en/stable/module_guides/storing/save_load.html#persisting-loading-data)). However, it's a bit of a different story when it comes to persisting the `ObjectNodeMapping`. Since we're indexing aribtrary Python objects with the `ObjectIndex`, it may be the case (and perhaps more often than we'd like), that the arbitrary objects are not serializable. In those cases, you can persist the index, but the user would have to maintain a way to re-construct the `ObjectNodeMapping` to be able to re-construct the `ObjectIndex`. For convenience, there are the `persist` and `from_persist_dir` methods on the `ObjectIndex` that will attempt to persist and load a previously saved `ObjectIndex`, respectively.

### Happy example
"""
logger.info("## Persisting `ObjectIndex` to Disk with Objects")

object_index.persist()

reloaded_object_index = ObjectIndex.from_persist_dir()

reloaded_object_index._object_node_mapping.obj_node_mapping

object_index._object_node_mapping.obj_node_mapping

"""
### Example of when it doesn't work
"""
logger.info("### Example of when it doesn't work")



def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

object_mapping = SimpleToolNodeMapping.from_objects([add_tool, multiply_tool])
object_index = ObjectIndex.from_objects(
    [add_tool, multiply_tool], object_mapping
)

object_mapping.persist()

object_index.persist()

"""
**In this case, only the index has been persisted.** In order to re-construct the `ObjectIndex` as mentioned above, we will need to manually re-construct `ObjectNodeMapping` and supply that to the `ObjectIndex.from_persist_dir` method.
"""

reloaded_object_index = ObjectIndex.from_persist_dir(
    object_node_mapping=object_mapping  # without this, an error will be thrown
)

logger.info("\n\n[DONE]", bright=True)