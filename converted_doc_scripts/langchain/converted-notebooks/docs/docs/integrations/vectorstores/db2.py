from jet.models.config import MODELS_CACHE_DIR
from jet.logger import logger
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_db2 import db2vs
from langchain_db2.db2vs import DB2VS
import ibm_db
import ibm_db_dbi
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
# IBM Db2 Vector Store and Vector Search

LangChain's Db2 integration (langchain-db2) provides vector store and vector search capabilities for working with IBM relational database Db2 version v12.1.2 and above, distributed under the MIT license. Users can use the provided implementations as-is or customize them for specific needs.
 Key features include:

 * Vector storage with metadata
 * Vector similarity search and max marginal relevance search, with metadata filtering options
 * Support for dot production, cosine, and euclidean distance metrics
 * Performance optimization by index creation and Approximate nearest neighbors search. (Will be added shortly)

## Setup

### Prerequisites for using Langchain with Db2 Vector Store and Search

Install package `langchain-db2` which is the integration package for the db2 LangChain Vector Store and Search.

The installation of the package should also install its dependencies like `langchain-core` and `ibm_db`.
"""
logger.info("# IBM Db2 Vector Store and Vector Search")



"""
### Connect to Db2 Vector Store

The following sample code will show how to connect to Db2 Database. Besides the dependencies above, you will need a Db2 database instance (with version v12.1.2+, which has the vector datatype support) running.
"""
logger.info("### Connect to Db2 Vector Store")


database = ""
username = ""
password = ""

try:
    connection = ibm_db_dbi.connect(database, username, password)
    logger.debug("Connection successful!")
except Exception as e:
    logger.debug("Connection failed!")

"""
### Import the required dependencies
"""
logger.info("### Import the required dependencies")


"""
## Initialization

### Create Documents
"""
logger.info("## Initialization")

documents_json_list = [
    {
        "id": "doc_1_2_P4",
        "text": "Db2 handles LOB data differently than other kinds of data. As a result, you sometimes need to take additional actions when you define LOB columns and insert the LOB data.",
        "link": "https://www.ibm.com/docs/en/db2-for-zos/12?topic=programs-storing-lob-data-in-tables",
    },
    {
        "id": "doc_11.1.0_P1",
        "text": "Db2® column-organized tables add columnar capabilities to Db2 databases, which include data that is stored with column organization and vector processing of column data. Using this table format with star schema data marts provides significant improvements to storage, query performance, and ease of use through simplified design and tuning.",
        "link": "https://www.ibm.com/docs/en/db2/11.1.0?topic=organization-column-organized-tables",
    },
    {
        "id": "id_22.3.4.3.1_P2",
        "text": "Data structures are elements that are required to use Db2®. You can access and use these elements to organize your data. Examples of data structures include tables, table spaces, indexes, index spaces, keys, views, and databases.",
        "link": "https://www.ibm.com/docs/en/zos-basic-skills?topic=concepts-db2-data-structures",
    },
    {
        "id": "id_3.4.3.1_P3",
        "text": "Db2® maintains a set of tables that contain information about the data that Db2 controls. These tables are collectively known as the catalog. The catalog tables contain information about Db2 objects such as tables, views, and indexes. When you create, alter, or drop an object, Db2 inserts, updates, or deletes rows of the catalog that describe the object.",
        "link": "https://www.ibm.com/docs/en/zos-basic-skills?topic=objects-db2-catalog",
    },
]

documents_langchain = []

for doc in documents_json_list:
    metadata = {"id": doc["id"], "link": doc["link"]}
    doc_langchain = Document(page_content=doc["text"], metadata=metadata)
    documents_langchain.append(doc_langchain)

"""
### Create Vector Stores with different distance metrics

First we will create three vector stores each with different distance strategies. 

(You can manually connect to the Db2 Database and will see three tables : 
Documents_DOT, Documents_COSINE and Documents_EUCLIDEAN. )
"""
logger.info("### Create Vector Stores with different distance metrics")

model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store_dot = DB2VS.from_documents(
    documents_langchain,
    model,
    client=connection,
    table_name="Documents_DOT",
    distance_strategy=DistanceStrategy.DOT_PRODUCT,
)
vector_store_max = DB2VS.from_documents(
    documents_langchain,
    model,
    client=connection,
    table_name="Documents_COSINE",
    distance_strategy=DistanceStrategy.COSINE,
)
vector_store_euclidean = DB2VS.from_documents(
    documents_langchain,
    model,
    client=connection,
    table_name="Documents_EUCLIDEAN",
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
)

"""
## Manage vector store

### Demonstrating add and delete operations for texts, along with basic similarity search
"""
logger.info("## Manage vector store")

def manage_texts(vector_stores):
    """
    Adds texts to each vector store, demonstrates error handling for duplicate additions,
    and performs deletion of texts. Showcases similarity searches and index creation for each vector store.

    Args:
    - vector_stores (list): A list of DB2VS instances.
    """
    texts = ["Rohan", "Shailendra"]
    metadata = [
        {"id": "100", "link": "Document Example Test 1"},
        {"id": "101", "link": "Document Example Test 2"},
    ]

    for i, vs in enumerate(vector_stores, start=1):
        try:
            vs.add_texts(texts, metadata)
            logger.debug(f"\n\n\nAdd texts complete for vector store {i}\n\n\n")
        except Exception as ex:
            logger.debug(f"\n\n\nExpected error on duplicate add for vector store {i}\n\n\n")

        vs.delete([metadata[0]["id"], metadata[1]["id"]])
        logger.debug(f"\n\n\nDelete texts complete for vector store {i}\n\n\n")

        results = vs.similarity_search("How are LOBS stored in Db2 Database", 2)
        logger.debug(f"\n\n\nSimilarity search results for vector store {i}: {results}\n\n\n")


vector_store_list = [
    vector_store_dot,
    vector_store_max,
    vector_store_euclidean,
]
manage_texts(vector_store_list)

"""
## Query vector store

### Demonstrate advanced searches on vector stores, with and without attribute filtering 

With filtering, we only select the document id 101 and nothing else
"""
logger.info("## Query vector store")

def conduct_advanced_searches(vector_stores):
    query = "How are LOBS stored in Db2 Database"
    filter_criteria = {"id": ["101"]}  # Direct comparison filter

    for i, vs in enumerate(vector_stores, start=1):
        logger.debug(f"\n--- Vector Store {i} Advanced Searches ---")
        logger.debug("\nSimilarity search results without filter:")
        logger.debug(vs.similarity_search(query, 2))

        logger.debug("\nSimilarity search results with filter:")
        logger.debug(vs.similarity_search(query, 2, filter=filter_criteria))

        logger.debug("\nSimilarity search with relevance score:")
        logger.debug(vs.similarity_search_with_score(query, 2))

        logger.debug("\nSimilarity search with relevance score with filter:")
        logger.debug(vs.similarity_search_with_score(query, 2, filter=filter_criteria))

        logger.debug("\nMax marginal relevance search results:")
        logger.debug(vs.max_marginal_relevance_search(query, 2, fetch_k=20, lambda_mult=0.5))

        logger.debug("\nMax marginal relevance search results with filter:")
        logger.debug(
            vs.max_marginal_relevance_search(
                query, 2, fetch_k=20, lambda_mult=0.5, filter=filter_criteria
            )
        )


conduct_advanced_searches(vector_store_list)

"""
## Usage for retrieval-augmented generation

## API reference
"""
logger.info("## Usage for retrieval-augmented generation")

logger.info("\n\n[DONE]", bright=True)