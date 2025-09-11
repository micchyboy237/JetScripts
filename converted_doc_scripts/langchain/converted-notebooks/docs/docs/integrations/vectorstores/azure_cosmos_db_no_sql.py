from azure.cosmos import CosmosClient, PartitionKey
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.azure_cosmos_db_no_sql import (
AzureCosmosDBNoSqlVectorSearch,
)
from langchain_community.vectorstores.azure_cosmos_db_no_sql import CosmosDBQueryType
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
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
# Azure Cosmos DB No SQL

This notebook shows you how to leverage this integrated [vector database](https://learn.microsoft.com/en-us/azure/cosmos-db/vector-database) to store documents in collections, create indicies and perform vector search queries using approximate nearest neighbor algorithms such as COS (cosine distance), L2 (Euclidean distance), and IP (inner product) to locate documents close to the query vectors.

Azure Cosmos DB is the database that powers Ollama's ChatGPT service. It offers single-digit millisecond response times, automatic and instant scalability, along with guaranteed speed at any scale.

Azure Cosmos DB for NoSQL now offers vector indexing and search in preview. This feature is designed to handle high-dimensional vectors, enabling efficient and accurate vector search at any scale. You can now store vectors directly in the documents alongside your data. This means that each document in your database can contain not only traditional schema-free data, but also high-dimensional vectors as other properties of the documents. This colocation of data and vectors allows for efficient indexing and searching, as the vectors are stored in the same logical unit as the data they represent. This simplifies data management, AI application architectures, and the efficiency of vector-based operations.

Please refer here for more details:
- [Vector Search](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/vector-search)
- [Full Text Search](https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/full-text-search)
- [Hybrid Search](https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/hybrid-search)

[Sign Up](https://azure.microsoft.com/en-us/free/) for lifetime free access to get started today.
"""
logger.info("# Azure Cosmos DB No SQL")

# %pip install --upgrade --quiet azure-cosmos langchain-ollama langchain-community

# OPENAI_API_KEY = "YOUR_KEY"
OPENAI_API_TYPE = "azure"
OPENAI_API_VERSION = "2023-05-15"
OPENAI_API_BASE = "YOUR_ENDPOINT"
OPENAI_EMBEDDINGS_MODEL_NAME = "text-embedding-ada-002"
OPENAI_EMBEDDINGS_MODEL_DEPLOYMENT = "text-embedding-ada-002"

"""
## Insert Data
"""
logger.info("## Insert Data")


loader = PyPDFLoader("https://arxiv.org/pdf/2303.08774.pdf")
data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(data)

logger.debug(docs[0])

"""
## Creating AzureCosmosDB NoSQL Vector Search
"""
logger.info("## Creating AzureCosmosDB NoSQL Vector Search")

indexing_policy = {
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}],
    "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
    "fullTextIndexes": [{"path": "/text"}],
}

vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": 1536,
        }
    ]
}

full_text_policy = {
    "defaultLanguage": "en-US",
    "fullTextPaths": [{"path": "/text", "language": "en-US"}],
}


HOST = "AZURE_COSMOS_DB_ENDPOINT"
KEY = "AZURE_COSMOS_DB_KEY"

cosmos_client = CosmosClient(HOST, KEY)
database_name = "langchain_python_db"
container_name = "langchain_python_container"
partition_key = PartitionKey(path="/id")
cosmos_container_properties = {"partition_key": partition_key}

ollama_embeddings = OllamaEmbeddings(
    deployment="smart-agent-embedding-ada",
    model="text-embedding-ada-002",
    chunk_size=1,
#     ollama_)

vector_search=AzureCosmosDBNoSqlVectorSearch.from_documents(
    documents=docs,
    embedding=ollama_embeddings,
    cosmos_client=cosmos_client,
    database_name=database_name,
    container_name=container_name,
    vector_embedding_policy=vector_embedding_policy,
    full_text_policy=full_text_policy,
    indexing_policy=indexing_policy,
    cosmos_container_properties=cosmos_container_properties,
    cosmos_database_properties={},
    full_text_search_enabled=True,
)

"""
#
#

V
e
c
t
o
r

S
e
a
r
c
h
"""
logger.info("#")


query="What were the compute requirements for training GPT 4"
results=vector_search.similarity_search(query)

logger.debug(results[0].page_content)

"""
#
#

V
e
c
t
o
r

S
e
a
r
c
h

w
i
t
h

S
c
o
r
e
"""
logger.info("#")

query="What were the compute requirements for training GPT 4"

results=vector_search.similarity_search_with_score(
    query=query,
    k=5,
)

for i in range(0, len(results)):
    logger.debug(f"Result {i + 1}: ", results[i][0].json())
    logger.debug(f"Score {i + 1}: ", results[i][1])
    logger.debug("\n")

"""
#
#

V
e
c
t
o
r

S
e
a
r
c
h

w
i
t
h

f
i
l
t
e
r
i
n
g
"""
logger.info("#")

query="What were the compute requirements for training GPT 4"

pre_filter={
    "conditions": [
        {"property": "metadata.page", "operator": "$eq", "value": 0},
    ],
}

results=vector_search.similarity_search_with_score(
    query=query,
    k=5,
    pre_filter=pre_filter,
)

for i in range(0, len(results)):
    logger.debug(f"Result {i + 1}: ", results[i][0].json())
    logger.debug(f"Score {i + 1}: ", results[i][1])
    logger.debug("\n")

"""
#
#

F
u
l
l

T
e
x
t

S
e
a
r
c
h
"""
logger.info("#")


query="What were the compute requirements for training GPT 4"
pre_filter={
    "conditions": [
        {
            "property": "text",
            "operator": "$full_text_contains_any",
            "value": "What were the compute requirements for training GPT 4",
        },
    ],
}
results=vector_search.similarity_search_with_score(
    query=query,
    k=5,
    query_type=CosmosDBQueryType.FULL_TEXT_SEARCH,
    pre_filter=pre_filter,
)

for i in range(0, len(results)):
    logger.debug(f"Result {i + 1}: ", results[i][0].json())
    logger.debug("\n")

"""
#
#

F
u
l
l

T
e
x
t

S
e
a
r
c
h

B
M

2
5

R
a
n
k
i
n
g
"""
logger.info("#")

query="What were the compute requirements for training GPT 4"

results=vector_search.similarity_search_with_score(
    query=query,
    k=5,
    query_type=CosmosDBQueryType.FULL_TEXT_RANK,
)

for i in range(0, len(results)):
    logger.debug(f"Result {i + 1}: ", results[i][0].json())
    logger.debug("\n")

"""
#
#

H
y
b
r
i
d

S
e
a
r
c
h
"""
logger.info("#")

query="What were the compute requirements for training GPT 4"

results=vector_search.similarity_search_with_score(
    query=query,
    k=5,
    query_type=CosmosDBQueryType.HYBRID,
)

for i in range(0, len(results)):
    logger.debug(f"Result {i + 1}: ", results[i][0].json())
    logger.debug(f"Score {i + 1}: ", results[i][1])
    logger.debug("\n")

"""
#
#

H
y
b
r
i
d

S
e
a
r
c
h

w
i
t
h

f
i
l
t
e
r
i
n
g
"""
logger.info("#")

query="What were the compute requirements for training GPT 4"

pre_filter={
    "conditions": [
        {
            "property": "text",
            "operator": "$full_text_contains_any",
            "value": "compute requirements",
        },
        {"property": "metadata.page", "operator": "$eq", "value": 0},
    ],
    "logical_operator": "$and",
}

results=vector_search.similarity_search_with_score(
    query=query, k=5, query_type=CosmosDBQueryType.HYBRID, pre_filter=pre_filter
)

for i in range(0, len(results)):
    logger.debug(f"Result {i + 1}: ", results[i][0].json())
    logger.debug(f"Score {i + 1}: ", results[i][1])
    logger.debug("\n")

logger.info("\n\n[DONE]", bright=True)
