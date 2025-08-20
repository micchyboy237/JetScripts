from IPython.display import Markdown, display
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
StorageContext,
SimpleDirectoryReader,
KnowledgeGraphIndex,
Settings,
)
from llama_index.core.settings import Settings
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neptune import (
NeptuneAnalyticsGraphStore,
NeptuneDatabaseGraphStore,
)
from llama_index.llms.bedrock import Bedrock
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
# Amazon Neptune Graph Store
"""
logger.info("# Amazon Neptune Graph Store")

# %pip install boto3
# %pip install llama-index-llms-bedrock
# %pip install llama-index-graph-stores-neptune
# %pip install llama-index-embeddings-bedrock

"""
## Using Knowledge Graph with NeptuneDatabaseGraphStore

### Add the required imports
"""
logger.info("## Using Knowledge Graph with NeptuneDatabaseGraphStore")


"""
### Configure the LLM to use, in this case Amazon Bedrock and Claude 2.1
"""
logger.info("### Configure the LLM to use, in this case Amazon Bedrock and Claude 2.1")

llm = Bedrock(model="anthropic.claude-v2")
embed_model = BedrockEmbedding(model="amazon.titan-embed-text-v1")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

"""
### Building the Knowledge Graph

### Read in the sample file
"""
logger.info("### Building the Knowledge Graph")

documents = SimpleDirectoryReader(
    "../../../../examples/paul_graham_essay/data"
).load_data()

"""
### Instantiate Neptune KG Indexes

When using Amazon Neptune you can choose to use either Neptune Database or Neptune Analytics.

Neptune Database is a serverless graph database designed for optimal scalability and availability. It provides a solution for graph database workloads that need to scale to 100,000 queries per second, Multi-AZ high availability, and multi-Region deployments. You can use Neptune Database for social networking, fraud alerting, and Customer 360 applications.

Neptune Analytics is an analytics database engine that can quickly analyze large amounts of graph data in memory to get insights and find trends. Neptune Analytics is a solution for quickly analyzing existing graph databases or graph datasets stored in a data lake. It uses popular graph analytic algorithms and low-latency analytic queries.


#### Using Neptune Database
If you can choose to use [Neptune Database](https://docs.aws.amazon.com/neptune/latest/userguide/feature-overview.html) to store your KG index you can create the graph store as shown below.
"""
logger.info("### Instantiate Neptune KG Indexes")

graph_store = NeptuneDatabaseGraphStore(
    host="<GRAPH NAME>.<CLUSTER ID>.<REGION>.neptune.amazonaws.com", port=8182
)

"""
#### Neptune Analytics

If you can choose to use [Neptune Analytics](https://docs.aws.amazon.com/neptune-analytics/latest/userguide/what-is-neptune-analytics.html) to store your KG index you can create the graph store as shown below.
"""
logger.info("#### Neptune Analytics")

graph_store = NeptuneAnalyticsGraphStore(
    graph_identifier="<INSERT GRAPH IDENIFIER>"
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
)

"""
#### Querying the Knowledge Graph

First, we can query and send only the triplets to the LLM.
"""
logger.info("#### Querying the Knowledge Graph")

query_engine = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)

response = query_engine.query("Tell me more about Interleaf")

display(Markdown(f"<b>{response}</b>"))

"""
For more detailed answers, we can also send the text from where the retrieved tripets were extracted.
"""
logger.info("For more detailed answers, we can also send the text from where the retrieved tripets were extracted.")

query_engine = index.as_query_engine(
    include_text=True, response_mode="tree_summarize"
)
response = query_engine.query(
    "Tell me more about what the author worked on at Interleaf"
)

display(Markdown(f"<b>{response}</b>"))

logger.info("\n\n[DONE]", bright=True)