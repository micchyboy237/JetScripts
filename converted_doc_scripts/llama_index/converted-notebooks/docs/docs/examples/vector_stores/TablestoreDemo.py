from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import MockEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import (
VectorStoreQuery,
MetadataFilters,
MetadataFilter,
FilterCondition,
FilterOperator,
)
from llama_index.core.vector_stores.types import (
VectorStoreQueryMode,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.tablestore import TablestoreVectorStore
from tablestore import FieldSchema, FieldType, VectorMetricType
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
# TablestoreVectorStore

> [Tablestore](https://www.aliyun.com/product/ots) is a fully managed NoSQL cloud database service that enables storage of a massive amount of structured
and semi-structured data.

This notebook shows how to use functionality related to the `Tablestore` vector database.

To use Tablestore, you must create an instance.
Here are the [creating instance instructions](https://help.aliyun.com/zh/tablestore/getting-started/manage-the-wide-column-model-in-the-tablestore-console).

## Install
"""
logger.info("# TablestoreVectorStore")

# %pip install llama-index-vector-stores-tablestore

# import getpass

# os.environ["end_point"] = getpass.getpass("Tablestore end_point:")
# os.environ["instance_name"] = getpass.getpass("Tablestore instance_name:")
# os.environ["access_key_id"] = getpass.getpass("Tablestore access_key_id:")
# os.environ["access_key_secret"] = getpass.getpass(
    "Tablestore access_key_secret:"
)

"""
## Example

Create vector store.
"""
logger.info("## Example")




vector_dimension = 4

store = TablestoreVectorStore(
    endpoint=os.getenv("end_point"),
    instance_name=os.getenv("instance_name"),
    access_key_id=os.getenv("access_key_id"),
    access_key_secret=os.getenv("access_key_secret"),
    vector_dimension=vector_dimension,
    vector_metric_type=VectorMetricType.VM_COSINE,
    metadata_mappings=[
        FieldSchema(
            "type", FieldType.KEYWORD, index=True, enable_sort_and_agg=True
        ),
        FieldSchema(
            "time", FieldType.LONG, index=True, enable_sort_and_agg=True
        ),
    ],
)

"""
Create table and index.
"""
logger.info("Create table and index.")

store.create_table_if_not_exist()
store.create_search_index_if_not_exist()

"""
New a mock embedding for test.
"""
logger.info("New a mock embedding for test.")

embedder = MockEmbedding(vector_dimension)

"""
Prepare some docs.
"""
logger.info("Prepare some docs.")

texts = [
    TextNode(
        id_="1",
        text="The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
        metadata={"type": "a", "time": 1995},
    ),
    TextNode(
        id_="2",
        text="When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
        metadata={"type": "a", "time": 1990},
    ),
    TextNode(
        id_="3",
        text="An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much, much more.",
        metadata={"type": "a", "time": 2009},
    ),
    TextNode(
        id_="4",
        text="A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into thed of a C.E.O.",
        metadata={"type": "a", "time": 2023},
    ),
    TextNode(
        id_="5",
        text="A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
        metadata={"type": "b", "time": 2018},
    ),
    TextNode(
        id_="6",
        text="Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives.",
        metadata={"type": "c", "time": 2010},
    ),
    TextNode(
        id_="7",
        text="An organized crime dynasty's aging patriarch transfers control of his clandestine empire to his reluctant son.",
        metadata={"type": "a", "time": 2023},
    ),
]
for t in texts:
    t.embedding = embedder.get_text_embedding(t.text)

"""
Write some docs.
"""
logger.info("Write some docs.")

store.add(texts)

"""
Delete docs.
"""
logger.info("Delete docs.")

store.delete("1")

"""
Query with filters.
"""
logger.info("Query with filters.")

store.query(
    query=VectorStoreQuery(
        query_embedding=embedder.get_text_embedding("nature fight physical"),
        similarity_top_k=5,
        filters=MetadataFilters(
            filters=[
                MetadataFilter(
                    key="type", value="a", operator=FilterOperator.EQ
                ),
                MetadataFilter(
                    key="time", value=2020, operator=FilterOperator.LTE
                ),
            ],
            condition=FilterCondition.AND,
        ),
    ),
)

"""
Full text search: query mode = TEXT.
"""
logger.info("Full text search: query mode = TEXT.")

query_result = store.query(
    query=VectorStoreQuery(
        mode=VectorStoreQueryMode.TEXT_SEARCH,
        query_str="computer",
        similarity_top_k=5,
    ),
)
logger.debug(query_result)

"""
HYBRID query.
"""
logger.info("HYBRID query.")

query_result = store.query(
    query=VectorStoreQuery(
        mode=VectorStoreQueryMode.HYBRID,
        query_embedding=embedder.get_text_embedding("nature fight physical"),
        query_str="python",
        similarity_top_k=5,
    ),
)
logger.debug(query_result)

logger.info("\n\n[DONE]", bright=True)