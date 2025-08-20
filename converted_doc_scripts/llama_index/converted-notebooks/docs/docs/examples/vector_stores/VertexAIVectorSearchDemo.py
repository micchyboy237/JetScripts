from google.cloud import aiplatform
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
StorageContext,
Settings,
VectorStoreIndex,
SimpleDirectoryReader,
)
from llama_index.core.schema import TextNode
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import (
MetadataFilters,
MetadataFilter,
FilterOperator,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.llms.vertex import Vertex
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/VertexAIVectorSearchDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Google Vertex AI Vector Search

This notebook shows how to use functionality related to the `Google Cloud Vertex AI Vector Search` vector database.

> [Google Vertex AI Vector Search](https://cloud.google.com/vertex-ai/docs/vector-search/overview), formerly known as Vertex AI Matching Engine, provides the industry's leading high-scale low latency vector database. These vector databases are commonly referred to as vector similarity-matching or an approximate nearest neighbor (ANN) service.

**Note**: LlamaIndex expects Vertex AI Vector Search endpoint and deployed index is already created. An empty index creation time take upto a minute and deploying an index to the endpoint can take upto 30 min.

> To see how to create an index refer to the section [Create Index and deploy it to an Endpoint](#create-index-and-deploy-it-to-an-endpoint)  
If you already have an index deployed , skip to [Create VectorStore from texts](#create-vector-store-from-texts)

## Installation

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Google Vertex AI Vector Search")

# ! pip install llama-index llama-index-vector-stores-vertexaivectorsearch llama-index-llms-vertex

"""
## Create Index and deploy it to an Endpoint

- This section demonstrates creating a new index and deploying it to an endpoint.
"""
logger.info("## Create Index and deploy it to an Endpoint")

PROJECT_ID = "[your_project_id]"
REGION = "[your_region]"
GCS_BUCKET_NAME = "[your_gcs_bucket]"
GCS_BUCKET_URI = f"gs://{GCS_BUCKET_NAME}"

VS_DIMENSIONS = 768

VS_INDEX_NAME = "llamaindex-doc-index"  # @param {type:"string"}
VS_INDEX_ENDPOINT_NAME = "llamaindex-doc-endpoint"  # @param {type:"string"}


aiplatform.init(project=PROJECT_ID, location=REGION)

"""
### Create Cloud Storage bucket
"""
logger.info("### Create Cloud Storage bucket")

# ! gsutil mb -l $REGION -p $PROJECT_ID $GCS_BUCKET_URI

"""
### Create an empty Index 

**Note :** While creating an index you should specify an "index_update_method" - `BATCH_UPDATE` or `STREAM_UPDATE`

> A batch index is for when you want to update your index in a batch, with data which has been stored over a set amount of time, like systems which are processed weekly or monthly. 
>
> A streaming index is when you want index data to be updated as new data is added to your datastore, for instance, if you have a bookstore and want to show new inventory online as soon as possible. 
>
> Which type you choose is important, since setup and requirements are different.

Refer [Official Documentation](https://cloud.google.com/vertex-ai/docs/vector-search/create-manage-index) and [API reference](https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.MatchingEngineIndex#google_cloud_aiplatform_MatchingEngineIndex_create_tree_ah_index) for more details on configuring indexes
"""
logger.info("### Create an empty Index")

index_names = [
    index.resource_name
    for index in aiplatform.MatchingEngineIndex.list(
        filter=f"display_name={VS_INDEX_NAME}"
    )
]

if len(index_names) == 0:
    logger.debug(f"Creating Vector Search index {VS_INDEX_NAME} ...")
    vs_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=VS_INDEX_NAME,
        dimensions=VS_DIMENSIONS,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
        shard_size="SHARD_SIZE_SMALL",
        index_update_method="STREAM_UPDATE",  # allowed values BATCH_UPDATE , STREAM_UPDATE
    )
    logger.debug(
        f"Vector Search index {vs_index.display_name} created with resource name {vs_index.resource_name}"
    )
else:
    vs_index = aiplatform.MatchingEngineIndex(index_name=index_names[0])
    logger.debug(
        f"Vector Search index {vs_index.display_name} exists with resource name {vs_index.resource_name}"
    )

"""
### Create an Endpoint

To use the index, you need to create an index endpoint. It works as a server instance accepting query requests for your index. An endpoint can be a [public endpoint](https://cloud.google.com/vertex-ai/docs/vector-search/deploy-index-public) or a [private endpoint](https://cloud.google.com/vertex-ai/docs/vector-search/deploy-index-vpc).

Let's create a public endpoint.
"""
logger.info("### Create an Endpoint")

endpoint_names = [
    endpoint.resource_name
    for endpoint in aiplatform.MatchingEngineIndexEndpoint.list(
        filter=f"display_name={VS_INDEX_ENDPOINT_NAME}"
    )
]

if len(endpoint_names) == 0:
    logger.debug(
        f"Creating Vector Search index endpoint {VS_INDEX_ENDPOINT_NAME} ..."
    )
    vs_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=VS_INDEX_ENDPOINT_NAME, public_endpoint_enabled=True
    )
    logger.debug(
        f"Vector Search index endpoint {vs_endpoint.display_name} created with resource name {vs_endpoint.resource_name}"
    )
else:
    vs_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=endpoint_names[0]
    )
    logger.debug(
        f"Vector Search index endpoint {vs_endpoint.display_name} exists with resource name {vs_endpoint.resource_name}"
    )

"""
### Deploy Index to the Endpoint

With the index endpoint, deploy the index by specifying a unique deployed index ID.

**NOTE : This operation can take upto 30 minutes.**
"""
logger.info("### Deploy Index to the Endpoint")

index_endpoints = [
    (deployed_index.index_endpoint, deployed_index.deployed_index_id)
    for deployed_index in vs_index.deployed_indexes
]

if len(index_endpoints) == 0:
    logger.debug(
        f"Deploying Vector Search index {vs_index.display_name} at endpoint {vs_endpoint.display_name} ..."
    )
    vs_deployed_index = vs_endpoint.deploy_index(
        index=vs_index,
        deployed_index_id=VS_INDEX_NAME,
        display_name=VS_INDEX_NAME,
        machine_type="e2-standard-16",
        min_replica_count=1,
        max_replica_count=1,
    )
    logger.debug(
        f"Vector Search index {vs_index.display_name} is deployed at endpoint {vs_deployed_index.display_name}"
    )
else:
    vs_deployed_index = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=index_endpoints[0][0]
    )
    logger.debug(
        f"Vector Search index {vs_index.display_name} is already deployed at endpoint {vs_deployed_index.display_name}"
    )

"""
## Create Vector Store from texts

NOTE : If you have existing Vertex AI Vector Search Index and Endpoints, you can assign them using following code:
"""
logger.info("## Create Vector Store from texts")

vs_index = aiplatform.MatchingEngineIndex(index_name="1234567890123456789")

vs_endpoint = aiplatform.MatchingEngineIndexEndpoint(
    index_endpoint_name="1234567890123456789"
)


"""
### Create a simple vector store from plain text without metadata filters
"""
logger.info("### Create a simple vector store from plain text without metadata filters")

vector_store = VertexAIVectorStore(
    project_id=PROJECT_ID,
    region=REGION,
    index_id=vs_index.resource_name,
    endpoint_id=vs_endpoint.resource_name,
    gcs_bucket_name=GCS_BUCKET_NAME,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

"""
### Use [Vertex AI Embeddings](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/embeddings/llama-index-embeddings-vertex) as the embeddings model
"""
logger.info("### Use [Vertex AI Embeddings](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/embeddings/llama-index-embeddings-vertex) as the embeddings model")

embed_model = VertexTextEmbedding(
    model_name="textembedding-gecko@003",
    project=PROJECT_ID,
    location=REGION,
)

Settings.embed_model = embed_model

"""
### Add vectors and mapped text chunks to your vectore store
"""
logger.info("### Add vectors and mapped text chunks to your vectore store")

texts = [
    "The cat sat on",
    "the mat.",
    "I like to",
    "eat pizza for",
    "dinner.",
    "The sun sets",
    "in the west.",
]
nodes = [
    TextNode(text=text, embedding=embed_model.get_text_embedding(text))
    for text in texts
]

vector_store.add(nodes)

"""
### Running a similarity search
"""
logger.info("### Running a similarity search")

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, embed_model=embed_model
)
retriever = index.as_retriever()

response = retriever.retrieve("pizza")
for row in response:
    logger.debug(f"Score: {row.get_score():.3f} Text: {row.get_text()}")

"""
## Add documents with metadata attributes and use filters
"""
logger.info("## Add documents with metadata attributes and use filters")

records = [
    {
        "description": "A versatile pair of dark-wash denim jeans."
        "Made from durable cotton with a classic straight-leg cut, these jeans"
        " transition easily from casual days to dressier occasions.",
        "price": 65.00,
        "color": "blue",
        "season": ["fall", "winter", "spring"],
    },
    {
        "description": "A lightweight linen button-down shirt in a crisp white."
        " Perfect for keeping cool with breathable fabric and a relaxed fit.",
        "price": 34.99,
        "color": "white",
        "season": ["summer", "spring"],
    },
    {
        "description": "A soft, chunky knit sweater in a vibrant forest green. "
        "The oversized fit and cozy wool blend make this ideal for staying warm "
        "when the temperature drops.",
        "price": 89.99,
        "color": "green",
        "season": ["fall", "winter"],
    },
    {
        "description": "A classic crewneck t-shirt in a soft, heathered blue. "
        "Made from comfortable cotton jersey, this t-shirt is a wardrobe essential "
        "that works for every season.",
        "price": 19.99,
        "color": "blue",
        "season": ["fall", "winter", "summer", "spring"],
    },
    {
        "description": "A flowing midi-skirt in a delicate floral print. "
        "Lightweight and airy, this skirt adds a touch of feminine style "
        "to warmer days.",
        "price": 45.00,
        "color": "white",
        "season": ["spring", "summer"],
    },
]

nodes = []
for record in records:
    text = record.pop("description")
    embedding = embed_model.get_text_embedding(text)
    metadata = {**record}
    nodes.append(TextNode(text=text, embedding=embedding, metadata=metadata))

vector_store.add(nodes)

"""
### Running a similarity search with filters
"""
logger.info("### Running a similarity search with filters")

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, embed_model=embed_model
)

retriever = index.as_retriever()
response = retriever.retrieve("pants")

for row in response:
    logger.debug(f"Text: {row.get_text()}")
    logger.debug(f"   Score: {row.get_score():.3f}")
    logger.debug(f"   Metadata: {row.metadata}")

filters = MetadataFilters(filters=[MetadataFilter(key="color", value="blue")])
retriever = index.as_retriever(filters=filters, similarity_top_k=3)
response = retriever.retrieve("denims")

for row in response:
    logger.debug(f"Text: {row.get_text()}")
    logger.debug(f"   Score: {row.get_score():.3f}")
    logger.debug(f"   Metadata: {row.metadata}")

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="color", value="blue"),
        MetadataFilter(key="price", operator=FilterOperator.GT, value=70.0),
    ]
)
retriever = index.as_retriever(filters=filters, similarity_top_k=3)
response = retriever.retrieve("denims")

for row in response:
    logger.debug(f"Text: {row.get_text()}")
    logger.debug(f"   Score: {row.get_score():.3f}")
    logger.debug(f"   Metadata: {row.metadata}")

"""
## Parse, Index and Query PDFs using Vertex AI Vector Search and Gemini Pro
"""
logger.info("## Parse, Index and Query PDFs using Vertex AI Vector Search and Gemini Pro")

# ! mkdir -p ./data/arxiv/
# ! wget 'https://arxiv.org/pdf/1706.03762.pdf' -O ./data/arxiv/test.pdf

documents = SimpleDirectoryReader("./data/arxiv/").load_data()
logger.debug(f"# of documents = {len(documents)}")

vector_store = VertexAIVectorStore(
    project_id=PROJECT_ID,
    region=REGION,
    index_id=vs_index.resource_name,
    endpoint_id=vs_endpoint.resource_name,
    gcs_bucket_name=GCS_BUCKET_NAME,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

embed_model = VertexTextEmbedding(
    model_name="textembedding-gecko@003",
    project=PROJECT_ID,
    location=REGION,
)

vertex_gemini = Vertex(
    model="gemini-pro",
    context_window=100000,
    temperature=0,
    additional_kwargs={},
)

Settings.llm = vertex_gemini
Settings.embed_model = embed_model

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

query_engine = index.as_query_engine()

response = query_engine.query(
    "who are the authors of paper Attention is All you need?"
)

logger.debug(f"Response:")
logger.debug("-" * 80)
logger.debug(response.response)
logger.debug("-" * 80)
logger.debug(f"Source Documents:")
logger.debug("-" * 80)
for source in response.source_nodes:
    logger.debug(f"Sample Text: {source.text[:50]}")
    logger.debug(f"Relevance score: {source.get_score():.3f}")
    logger.debug(f"File Name: {source.metadata.get('file_name')}")
    logger.debug(f"Page #: {source.metadata.get('page_label')}")
    logger.debug(f"File Path: {source.metadata.get('file_path')}")
    logger.debug("-" * 80)

"""
---

## Clean Up

Please delete Vertex AI Vector Search Index and Index Endpoint after running your experiments to avoid incurring additional charges. Please note that you will be charged as long as the endpoint is running.

<div class="alert alert-block alert-warning">
    <b>⚠️ NOTE: Enabling `CLEANUP_RESOURCES` flag deletes Vector Search Index, Index Endpoint and Cloud Storage bucket. Please run it with caution.</b>
</div>
"""
logger.info("## Clean Up")

CLEANUP_RESOURCES = False

"""
- Undeploy indexes and Delete index endpoint
"""

if CLEANUP_RESOURCES:
    logger.debug(
        f"Undeploying all indexes and deleting the index endpoint {vs_endpoint.display_name}"
    )
    vs_endpoint.undeploy_all()
    vs_endpoint.delete()

"""
- Delete index
"""

if CLEANUP_RESOURCES:
    logger.debug(f"Deleting the index {vs_index.display_name}")
    vs_index.delete()

"""
- Delete contents from the Cloud Storage bucket
"""

if CLEANUP_RESOURCES and "GCS_BUCKET_NAME" in globals():
    logger.debug(f"Deleting contents from the Cloud Storage bucket {GCS_BUCKET_NAME}")

    shell_output = ! gsutil du -ash gs://$GCS_BUCKET_NAME
    logger.debug(shell_output)
    logger.debug(
        f"Size of the bucket {GCS_BUCKET_NAME} before deleting = {' '.join(shell_output[0].split()[:2])}"
    )

logger.info("\n\n[DONE]", bright=True)