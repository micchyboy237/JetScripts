from jet.models.config import MODELS_CACHE_DIR
from IPython.display import Markdown, display
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from jet.logger import CustomLogger
from llama_index.core import (
SimpleDirectoryReader,
StorageContext,
VectorStoreIndex,
)
from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import (
MetadataFilters,
MetadataFilter,
FilterOperator,
FilterCondition,
)
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.azure_openai import AzureHuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOllamaFunctionCallingAdapter
from llama_index.vector_stores.azureaisearch import (
IndexManagement,
MetadataIndexFieldType,
)
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
import logging
import os
import shutil
import sys
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/AzureAISearchIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Azure AI Search

## Basic Example

In this notebook, we take a Paul Graham essay, split it into chunks, embed it using an Azure OllamaFunctionCallingAdapter embedding model, load it into an Azure AI Search index, and then query it.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Azure AI Search")

# !pip install llama-index
# !pip install wget
# %pip install llama-index-vector-stores-azureaisearch
# %pip install azure-search-documents==11.5.1
# %llama-index-embeddings-azure-openai
# %llama-index-llms-azure-openai


"""
## Setup Azure OllamaFunctionCallingAdapter
"""
logger.info("## Setup Azure OllamaFunctionCallingAdapter")

# aoai_api_key = "YOUR_AZURE_OPENAI_API_KEY"
aoai_endpoint = "YOUR_AZURE_OPENAI_ENDPOINT"
aoai_api_version = "2024-10-21"

llm = AzureOllamaFunctionCallingAdapter(
    model="YOUR_AZURE_OPENAI_COMPLETION_MODEL_NAME",
    deployment_name="YOUR_AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME",
    api_key=aoai_api_key,
    azure_endpoint=aoai_endpoint,
    api_version=aoai_api_version,
)

embed_model = AzureHuggingFaceEmbedding(
    model="YOUR_AZURE_OPENAI_EMBEDDING_MODEL_NAME",
    deployment_name="YOUR_AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
    api_key=aoai_api_key,
    azure_endpoint=aoai_endpoint,
    api_version=aoai_api_version,
)

"""
## Setup Azure AI Search
"""
logger.info("## Setup Azure AI Search")

search_service_api_key = "YOUR-AZURE-SEARCH-SERVICE-ADMIN-KEY"
search_service_endpoint = "YOUR-AZURE-SEARCH-SERVICE-ENDPOINT"
search_service_api_version = "2024-07-01"
credential = AzureKeyCredential(search_service_api_key)


index_name = "llamaindex-vector-demo"

index_client = SearchIndexClient(
    endpoint=search_service_endpoint,
    credential=credential,
)

search_client = SearchClient(
    endpoint=search_service_endpoint,
    index_name=index_name,
    credential=credential,
)

"""
## Create Index (if it does not exist)

Demonstrates creating a vector index named "llamaindex-vector-demo" if one doesn't exist. The index has the following fields:
| Field Name | OData Type                |  
|------------|---------------------------|  
| id         | `Edm.String`              |  
| chunk      | `Edm.String`              |  
| embedding  | `Collection(Edm.Single)`  |  
| metadata   | `Edm.String`              |  
| doc_id     | `Edm.String`              |  
| author     | `Edm.String`              |  
| theme      | `Edm.String`              |  
| director   | `Edm.String`              |
"""
logger.info("## Create Index (if it does not exist)")

metadata_fields = {
    "author": "author",
    "theme": ("topic", MetadataIndexFieldType.STRING),
    "director": "director",
}

vector_store = AzureAISearchVectorStore(
    search_or_index_client=index_client,
    filterable_metadata_field_keys=metadata_fields,
    index_name=index_name,
    index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
    id_field_key="id",
    chunk_field_key="chunk",
    embedding_field_key="embedding",
    embedding_dimensionality=1536,
    metadata_string_field_key="metadata",
    doc_id_field_key="doc_id",
    language_analyzer="en.lucene",
    vector_algorithm_type="exhaustiveKnn",
)

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
### Loading documents
Load the documents stored in the `data/paul_graham/` using the SimpleDirectoryReader
"""
logger.info("### Loading documents")

documents = SimpleDirectoryReader("./Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()
storage_context = StorageContext.from_defaults(vector_store=vector_store)

Settings.llm = llm
Settings.embed_model = embed_model
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("What did the author do growing up?")
display(Markdown(f"<b>{response}</b>"))

response = query_engine.query(
    "What did the author learn?",
)
display(Markdown(f"<b>{response}</b>"))

"""
## Use Existing Index
"""
logger.info("## Use Existing Index")

index_name = "llamaindex-vector-demo"

metadata_fields = {
    "author": "author",
    "theme": ("topic", MetadataIndexFieldType.STRING),
    "director": "director",
}
vector_store = AzureAISearchVectorStore(
    search_or_index_client=search_client,
    filterable_metadata_field_keys=metadata_fields,
    index_management=IndexManagement.VALIDATE_INDEX,
    id_field_key="id",
    chunk_field_key="chunk",
    embedding_field_key="embedding",
    embedding_dimensionality=1536,
    metadata_string_field_key="metadata",
    doc_id_field_key="doc_id",
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    [],
    storage_context=storage_context,
)

query_engine = index.as_query_engine()
response = query_engine.query("What was a hard moment for the author?")
display(Markdown(f"<b>{response}</b>"))

response = query_engine.query("Who is the author?")
display(Markdown(f"<b>{response}</b>"))


query_engine = index.as_query_engine(streaming=True)
response = query_engine.query("What happened at interleaf?")

start_time = time.time()

token_count = 0
for token in response.response_gen:
    logger.debug(token, end="")
    token_count += 1

time_elapsed = time.time() - start_time
tokens_per_second = token_count / time_elapsed

logger.debug(f"\n\nStreamed output at {tokens_per_second} tokens/s")

"""
## Adding a document to existing index
"""
logger.info("## Adding a document to existing index")

response = query_engine.query("What colour is the sky?")
display(Markdown(f"<b>{response}</b>"))


index.insert_nodes([Document(text="The sky is indigo today")])

response = query_engine.query("What colour is the sky?")
display(Markdown(f"<b>{response}</b>"))

"""
## Filtering

Filters can be applied to queries using either the `filters` parameter to use llama-index's filter syntax or the `odata_filters` parameter to pass in filters directly.
"""
logger.info("## Filtering")


nodes = [
    TextNode(
        text="The Shawshank Redemption",
        metadata={
            "author": "Stephen King",
            "theme": "Friendship",
        },
    ),
    TextNode(
        text="The Godfather",
        metadata={
            "director": "Francis Ford Coppola",
            "theme": "Mafia",
        },
    ),
    TextNode(
        text="Inception",
        metadata={
            "director": "Christopher Nolan",
        },
    ),
]

index.insert_nodes(nodes)



filters = MetadataFilters(
    filters=[
        MetadataFilter(key="theme", value="Mafia", operator=FilterOperator.EQ)
    ],
)

retriever = index.as_retriever(filters=filters)
retriever.retrieve("What is inception about?")

"""
Or passing in the odata_filters parameter directly:
"""
logger.info("Or passing in the odata_filters parameter directly:")

odata_filters = "theme eq 'Mafia'"
retriever = index.as_retriever(
    vector_store_kwargs={"odata_filters": odata_filters}
)
retriever.retrieve("What is inception about?")

"""
## Query Mode
Four query modes are supported: DEFAULT (vector search), SPARSE, HYBRID, and SEMANTIC_HYBRID.

### Perform a Vector Search
"""
logger.info("## Query Mode")


default_retriever = index.as_retriever(
    vector_store_query_mode=VectorStoreQueryMode.DEFAULT
)
response = default_retriever.retrieve("What is inception about?")

for node_with_score in response:
    node = node_with_score.node  # The TextNode object
    score = node_with_score.score  # The similarity score
    chunk_id = node.id_  # The chunk ID

    file_name = node.metadata.get("file_name", "Unknown")
    file_path = node.metadata.get("file_path", "Unknown")

    text_content = node.text if node.text else "No content available"

    logger.debug(f"Score: {score}")
    logger.debug(f"File Name: {file_name}")
    logger.debug(f"Id: {chunk_id}")
    logger.debug("\nExtracted Content:")
    logger.debug(text_content)
    logger.debug("\n" + "=" * 40 + " End of Result " + "=" * 40 + "\n")

"""
### Perform a Hybrid Search
"""
logger.info("### Perform a Hybrid Search")


hybrid_retriever = index.as_retriever(
    vector_store_query_mode=VectorStoreQueryMode.HYBRID
)
hybrid_retriever.retrieve("What is inception about?")

"""
### Perform a Hybrid Search with Semantic Reranking
This mode incorporates semantic reranking to hybrid search results to improve search relevance. 

Please see this link for further details: https://learn.microsoft.com/azure/search/semantic-search-overview
"""
logger.info("### Perform a Hybrid Search with Semantic Reranking")

hybrid_retriever = index.as_retriever(
    vector_store_query_mode=VectorStoreQueryMode.SEMANTIC_HYBRID
)
hybrid_retriever.retrieve("What is inception about?")

logger.info("\n\n[DONE]", bright=True)