from azure.search.documents.indexes.models import (
FreshnessScoringFunction,
FreshnessScoringParameters,
ScoringProfile,
SearchableField,
SearchField,
SearchFieldDataType,
SimpleField,
TextWeights,
)
from azure.search.documents.indexes.models import (
ScoringProfile,
SearchableField,
SearchField,
SearchFieldDataType,
SimpleField,
TextWeights,
)
from datetime import datetime, timedelta
from jet.adapters.langchain.chat_ollama import AzureOllamaEmbeddings, OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_text_splitters import CharacterTextSplitter
from pprint import pprint
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
# Azure AI Search

[Azure AI Search](https://learn.microsoft.com/azure/search/search-what-is-azure-search) (formerly known as `Azure Search` and `Azure Cognitive Search`) is a cloud search service that gives developers infrastructure, APIs, and tools for information retrieval of vector, keyword, and hybrid queries at scale.

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

## Install Azure AI Search SDK

Use azure-search-documents package version 11.4.0 or later.
"""
logger.info("# Azure AI Search")

# %pip install --upgrade --quiet  azure-search-documents
# %pip install --upgrade --quiet  azure-identity

"""
## Import required libraries

`OllamaEmbeddings` is assumed, but if you're using Azure Ollama, import `AzureOllamaEmbeddings` instead.
"""
logger.info("## Import required libraries")



"""
## Configure Ollama settings
Set variables for your Ollama provider. You need either an [Ollama account](https://platform.ollama.com/docs/quickstart?context=python) or an [Azure Ollama account](https://learn.microsoft.com/en-us/azure/ai-services/ollama/how-to/create-resource) to generate the embeddings.
"""
logger.info("## Configure Ollama settings")

ollama_api_key: str = "PLACEHOLDER FOR YOUR API KEY"
ollama_api_version: str = "2023-05-15"
model: str = "text-embedding-ada-002"

azure_endpoint: str = "PLACEHOLDER FOR YOUR AZURE OPENAI ENDPOINT"
azure_ollama_api_key: str = "PLACEHOLDER FOR YOUR AZURE OPENAI KEY"
azure_ollama_api_version: str = "2023-05-15"
azure_deployment: str = "text-embedding-ada-002"

"""
## Configure vector store settings

You need an [Azure subscription](https://azure.microsoft.com/en-us/free/search) and [Azure AI Search service](https://learn.microsoft.com/azure/search/search-create-service-portal) to use this vector store integration. No-cost versions are available for small and limited workloads.
 
Set variables for your Azure AI Search URL and admin API key. You can get these variables from the [Azure portal](https://portal.azure.com/#blade/HubsExtension/BrowseResourceBlade/resourceType/Microsoft.Search%2FsearchServices).
"""
logger.info("## Configure vector store settings")

vector_store_address: str = "YOUR_AZURE_SEARCH_ENDPOINT"
vector_store_password: str = "YOUR_AZURE_SEARCH_ADMIN_KEY"

"""
## Create embeddings and vector store instances
 
Create instances of the OllamaEmbeddings and AzureSearch classes. When you complete this step, you should have an empty search index on your Azure AI Search resource. The integration module provides a default schema.
"""
logger.info("## Create embeddings and vector store instances")

embeddings: OllamaEmbeddings = OllamaEmbeddings(
    ollama_api_key=ollama_api_key, ollama_api_version=ollama_api_version, model=model
)

embeddings: AzureOllamaEmbeddings = AzureOllamaEmbeddings(
    azure_deployment=azure_deployment,
    ollama_api_version=azure_ollama_api_version,
    azure_endpoint=azure_endpoint,
    api_key=azure_ollama_api_key,
)

"""
## Create vector store instance
 
Create instance of the AzureSearch class using the embeddings from above
"""
logger.info("## Create vector store instance")

index_name: str = "langchain-vector-demo"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
    additional_search_client_options={"retry_total": 4},
)

"""
## Insert text and embeddings into vector store
 
This step loads, chunks, and vectorizes the sample document, and then indexes the content into a search index on Azure AI Search.
"""
logger.info("## Insert text and embeddings into vector store")


loader = TextLoader("../../how_to/state_of_the_union.txt", encoding="utf-8")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

vector_store.add_documents(documents=docs)

"""
## Perform a vector similarity search
 
Execute a pure vector similarity search using the similarity_search() method:
"""
logger.info("## Perform a vector similarity search")

docs = vector_store.similarity_search(
    query="What did the president say about Ketanji Brown Jackson",
    k=3,
    search_type="similarity",
)
logger.debug(docs[0].page_content)

"""
## Perform a vector similarity search with relevance scores
 
Execute a pure vector similarity search using the similarity_search_with_relevance_scores() method. Queries that don't meet the threshold requirements are exluded.
"""
logger.info("## Perform a vector similarity search with relevance scores")

docs_and_scores = vector_store.similarity_search_with_relevance_scores(
    query="What did the president say about Ketanji Brown Jackson",
    k=4,
    score_threshold=0.80,
)

plogger.debug(docs_and_scores)

"""
## Perform a hybrid search

Execute hybrid search using the search_type or hybrid_search() method. Vector and nonvector text fields are queried in parallel, results are merged, and top matches of the unified result set are returned.
"""
logger.info("## Perform a hybrid search")

docs = vector_store.similarity_search(
    query="What did the president say about Ketanji Brown Jackson",
    k=3,
    search_type="hybrid",
)
logger.debug(docs[0].page_content)

docs = vector_store.hybrid_search(
    query="What did the president say about Ketanji Brown Jackson", k=3
)
logger.debug(docs[0].page_content)

"""
## Custom schemas and queries

This section shows you how to replace the default schema with a custom schema.

### Create a new index with custom filterable fields 

This schema shows field definitions. It's the default schema, plus several new fields attributed as filterable. Because it's using the default vector configuration, you won't see vector configuration or vector profile overrides here. The name of the default vector profile is "myHnswProfile" and it's using a vector configuration of Hierarchical Navigable Small World (HNSW) for indexing and queries against the content_vector field.

There's no data for this schema in this step. When you execute the cell, you should get an empty index on Azure AI Search.
"""
logger.info("## Custom schemas and queries")


embeddings: OllamaEmbeddings = OllamaEmbeddings(
    ollama_api_key=ollama_api_key, ollama_api_version=ollama_api_version, model=model
)
embedding_function = embeddings.embed_query

fields = [
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        filterable=True,
    ),
    SearchableField(
        name="content",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=len(embedding_function("Text")),
        vector_search_profile_name="myHnswProfile",
    ),
    SearchableField(
        name="metadata",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    SearchableField(
        name="title",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    SimpleField(
        name="source",
        type=SearchFieldDataType.String,
        filterable=True,
    ),
]

index_name: str = "langchain-vector-demo-custom"

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embedding_function,
    fields=fields,
)

"""
### Add data and perform a query that includes a filter

This example adds data to the vector store based on the custom schema. It loads text into the title and source fields. The source field is filterable. The sample query in this section filters the results based on content in the source field.
"""
logger.info("### Add data and perform a query that includes a filter")

vector_store.add_texts(
    ["Test 1", "Test 2", "Test 3"],
    [
        {"title": "Title 1", "source": "A", "random": "10290"},
        {"title": "Title 2", "source": "A", "random": "48392"},
        {"title": "Title 3", "source": "B", "random": "32893"},
    ],
)

res = vector_store.similarity_search(query="Test 3 source1", k=3, search_type="hybrid")
res

res = vector_store.similarity_search(
    query="Test 3 source1", k=3, search_type="hybrid", filters="source eq 'A'"
)
res

"""
### Create a new index with a scoring profile

Here's another custom schema that includes a scoring profile definition. A scoring profile is used for relevance tuning of nonvector content, which is helpful in hybrid search scenarios.
"""
logger.info("### Create a new index with a scoring profile")


embeddings: OllamaEmbeddings = OllamaEmbeddings(
    ollama_api_key=ollama_api_key, ollama_api_version=ollama_api_version, model=model
)
embedding_function = embeddings.embed_query

fields = [
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        filterable=True,
    ),
    SearchableField(
        name="content",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=len(embedding_function("Text")),
        vector_search_profile_name="myHnswProfile",
    ),
    SearchableField(
        name="metadata",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    SearchableField(
        name="title",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    SimpleField(
        name="source",
        type=SearchFieldDataType.String,
        filterable=True,
    ),
    SimpleField(
        name="last_update",
        type=SearchFieldDataType.DateTimeOffset,
        searchable=True,
        filterable=True,
    ),
]
sc_name = "scoring_profile"
sc = ScoringProfile(
    name=sc_name,
    text_weights=TextWeights(weights={"title": 5}),
    function_aggregation="sum",
    functions=[
        FreshnessScoringFunction(
            field_name="last_update",
            boost=100,
            parameters=FreshnessScoringParameters(boosting_duration="P2D"),
            interpolation="linear",
        )
    ],
)

index_name = "langchain-vector-demo-custom-scoring-profile"

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
    fields=fields,
    scoring_profiles=[sc],
    default_scoring_profile=sc_name,
)


today = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S-00:00")
yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S-00:00")
one_month_ago = (datetime.utcnow() - timedelta(days=30)).strftime(
    "%Y-%m-%dT%H:%M:%S-00:00"
)

vector_store.add_texts(
    ["Test 1", "Test 1", "Test 1"],
    [
        {
            "title": "Title 1",
            "source": "source1",
            "random": "10290",
            "last_update": today,
        },
        {
            "title": "Title 1",
            "source": "source1",
            "random": "48392",
            "last_update": yesterday,
        },
        {
            "title": "Title 1",
            "source": "source1",
            "random": "32893",
            "last_update": one_month_ago,
        },
    ],
)

res = vector_store.similarity_search(query="Test 1", k=3, search_type="similarity")
res

logger.info("\n\n[DONE]", bright=True)