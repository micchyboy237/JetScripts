from jet.adapters.langchain.chat_ollama import AzureOllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.azure_cosmos_db import (
AzureCosmosDBVectorSearch,
CosmosDBSimilarityType,
CosmosDBVectorSearchType,
)
from langchain_text_splitters import CharacterTextSplitter
from pymongo import MongoClient
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
# Azure Cosmos DB Mongo vCore

This notebook shows you how to leverage this integrated [vector database](https://learn.microsoft.com/en-us/azure/cosmos-db/vector-database) to store documents in collections, create indicies and perform vector search queries using approximate nearest neighbor algorithms such as COS (cosine distance), L2 (Euclidean distance), and IP (inner product) to locate documents close to the query vectors. 
    
Azure Cosmos DB is the database that powers Ollama's ChatGPT service. It offers single-digit millisecond response times, automatic and instant scalability, along with guaranteed speed at any scale. 

Azure Cosmos DB for MongoDB vCore(https://learn.microsoft.com/en-us/azure/cosmos-db/mongodb/vcore/) provides developers with a fully managed MongoDB-compatible database service for building modern applications with a familiar architecture. You can apply your MongoDB experience and continue to use your favorite MongoDB drivers, SDKs, and tools by pointing your application to the API for MongoDB vCore account's connection string.

[Sign Up](https://azure.microsoft.com/en-us/free/) for lifetime free access to get started today.


"""
logger.info("# Azure Cosmos DB Mongo vCore")

# %pip install --upgrade --quiet  pymongo langchain-ollama langchain-community


CONNECTION_STRING = "YOUR_CONNECTION_STRING"
INDEX_NAME = "izzy-test-index"
NAMESPACE = "izzy_test_db.izzy_test_collection"
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")

"""
We want to use `AzureOllamaEmbeddings` so we need to set up our Azure Ollama API Key alongside other environment variables.
"""
logger.info("We want to use `AzureOllamaEmbeddings` so we need to set up our Azure Ollama API Key alongside other environment variables.")

# os.environ["AZURE_OPENAI_API_KEY"] = "YOUR_AZURE_OPENAI_API_KEY"
os.environ["AZURE_OPENAI_ENDPOINT"] = "YOUR_AZURE_OPENAI_ENDPOINT"
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_EMBEDDINGS_MODEL_NAME"] = "text-embedding-ada-002"  # the model name

"""
Now, we need to load the documents into the collection, create the index and then run our queries against the index to retrieve matches.

Please refer to the [documentation](https://learn.microsoft.com/en-us/azure/cosmos-db/mongodb/vcore/vector-search) if you have questions about certain parameters
"""
logger.info("Now, we need to load the documents into the collection, create the index and then run our queries against the index to retrieve matches.")


SOURCE_FILE_NAME = "../../how_to/state_of_the_union.txt"

loader = TextLoader(SOURCE_FILE_NAME)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

model_deployment = os.getenv(
    "OPENAI_EMBEDDINGS_DEPLOYMENT", "smart-agent-embedding-ada"
)
model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002")


ollama_embeddings: AzureOllamaEmbeddings = AzureOllamaEmbeddings(
    model=model_name, chunk_size=1
)

docs[0]



client: MongoClient = MongoClient(CONNECTION_STRING)
collection = client[DB_NAME][COLLECTION_NAME]

model_deployment = os.getenv(
    "OPENAI_EMBEDDINGS_DEPLOYMENT", "smart-agent-embedding-ada"
)
model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002")

vectorstore = AzureCosmosDBVectorSearch.from_documents(
    docs,
    ollama_embeddings,
    collection=collection,
    index_name=INDEX_NAME,
)

num_lists = 100
dimensions = 1536
similarity_algorithm = CosmosDBSimilarityType.COS
kind = CosmosDBVectorSearchType.VECTOR_IVF
m = 16
ef_construction = 64
ef_search = 40
score_threshold = 0.1

vectorstore.create_index(
    num_lists, dimensions, similarity_algorithm, kind, m, ef_construction
)

"""
maxDegree = 40
dimensions = 1536
similarity_algorithm = CosmosDBSimilarityType.COS
kind = CosmosDBVectorSearchType.VECTOR_DISKANN
lBuild = 20

vectorstore.create_index(
            dimensions=dimensions,
            similarity=similarity_algorithm,
            kind=kind ,
            max_degree=maxDegree,
            l_build=lBuild,
        )


dimensions = 1536
similarity_algorithm = CosmosDBSimilarityType.COS
kind = CosmosDBVectorSearchType.VECTOR_HNSW
m = 16
ef_construction = 64

vectorstore.create_index(
            dimensions=dimensions,
            similarity=similarity_algorithm,
            kind=kind ,
            m=m,
            ef_construction=ef_construction,
        )
"""

query = "What did the president say about Ketanji Brown Jackson"
docs = vectorstore.similarity_search(query)

logger.debug(docs[0].page_content)

"""
Once the documents have been loaded and the index has been created, you can now instantiate the vector store directly and run queries against the index
"""
logger.info("Once the documents have been loaded and the index has been created, you can now instantiate the vector store directly and run queries against the index")

vectorstore = AzureCosmosDBVectorSearch.from_connection_string(
    CONNECTION_STRING, NAMESPACE, ollama_embeddings, index_name=INDEX_NAME
)

query = "What did the president say about Ketanji Brown Jackson"
docs = vectorstore.similarity_search(query)

logger.debug(docs[0].page_content)

vectorstore = AzureCosmosDBVectorSearch(
    collection, ollama_embeddings, index_name=INDEX_NAME
)

query = "What did the president say about Ketanji Brown Jackson"
docs = vectorstore.similarity_search(query)

logger.debug(docs[0].page_content)

"""
## Filtered vector search (Preview)
Azure Cosmos DB for MongoDB supports pre-filtering with $lt, $lte, $eq, $neq, $gte, $gt, $in, $nin, and $regex. To use this feature, enable "filtering vector search" in the "Preview Features" tab of your Azure Subscription. Learn more about preview features [here](https://learn.microsoft.com/azure/cosmos-db/mongodb/vcore/vector-search#filtered-vector-search-preview).
"""
logger.info("## Filtered vector search (Preview)")

vectorstore.create_filter_index(
    property_to_filter="metadata.source", index_name="filter_index"
)

query = "What did the president say about Ketanji Brown Jackson"
docs = vectorstore.similarity_search(
    query, pre_filter={"metadata.source": {"$ne": "filter content"}}
)

len(docs)

docs = vectorstore.similarity_search(
    query,
    pre_filter={"metadata.source": {"$ne": "../../how_to/state_of_the_union.txt"}},
)

len(docs)

logger.info("\n\n[DONE]", bright=True)