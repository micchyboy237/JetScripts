from llama_index.core.indices.property_graph import CypherTemplateRetriever
from pydantic import BaseModel, Field
from llama_index.core.indices.property_graph import TextToCypherRetriever
import nest_asyncio
from IPython.display import Markdown, display
from llama_index.graph_stores.neptune import (
    NeptuneAnalyticsPropertyGraphStore,
    NeptuneDatabasePropertyGraphStore,
)
from llama_index.core import (
    StorageContext,
    SimpleDirectoryReader,
    PropertyGraphIndex,
    Settings,
)
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.bedrock import Bedrock
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# Amazon Neptune Property Graph Store

# %pip install boto3 nest_asyncio
# %pip install llama-index-llms-bedrock
# %pip install llama-index-graph-stores-neptune
# %pip install llama-index-embeddings-bedrock

# Using Property Graph with Amazon Neptune

# Add the required imports


# Configure the LLM to use, in this case Amazon Bedrock and Claude 2.1

llm = Bedrock(model="anthropic.claude-3-sonnet-20240229-v1:0")
embed_model = BedrockEmbedding(model="amazon.titan-embed-text-v2:0")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

# Building the Graph
#
# Read in the sample file


nest_asyncio.apply()

documents = SimpleDirectoryReader(
    "../../../../examples/paul_graham_essay/data"
).load_data()

# Instantiate Neptune Property Graph Indexes
#
# When using Amazon Neptune you can choose to use either Neptune Database or Neptune Analytics.
#
# Neptune Database is a serverless graph database designed for optimal scalability and availability. It provides a solution for graph database workloads that need to scale to 100,000 queries per second, Multi-AZ high availability, and multi-Region deployments. You can use Neptune Database for social networking, fraud alerting, and Customer 360 applications.
#
# Neptune Analytics is an analytics database engine that can quickly analyze large amounts of graph data in memory to get insights and find trends. Neptune Analytics is a solution for quickly analyzing existing graph databases or graph datasets stored in a data lake. It uses popular graph analytic algorithms and low-latency analytic queries.
#
#
# Using Neptune Database
# If you can choose to use [Neptune Database](https://docs.aws.amazon.com/neptune/latest/userguide/feature-overview.html) to store your property graph index you can create the graph store as shown below.

graph_store = NeptuneDatabasePropertyGraphStore(
    host="<GRAPH NAME>.<CLUSTER ID>.<REGION>.neptune.amazonaws.com", port=8182
)

# Neptune Analytics
#
# If you can choose to use [Neptune Analytics](https://docs.aws.amazon.com/neptune-analytics/latest/userguide/what-is-neptune-analytics.html) to store your property index you can create the graph store as shown below.

graph_store = NeptuneAnalyticsPropertyGraphStore(
    graph_identifier="<INSERT GRAPH IDENIFIER>"
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    storage_context=storage_context
)

index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store
)

# Querying the Property Graph
#
# First, we can query and send only the values to the LLM.

query_engine = index.as_query_engine(
    include_text=True,
    llm=llm,
)

response = query_engine.query("Tell me more about Interleaf")

display(Markdown(f"<b>{response}</b>"))

# Second, we can use the query using a retriever

retriever = index.as_retriever(
    include_text=True,
)

nodes = retriever.retrieve("What happened at Interleaf and Viaweb?")

# Third, we can use a `TextToCypherRetriever` to convert natural language questions into dynamic openCypher queries


cypher_retriever = TextToCypherRetriever(index.property_graph_store)

nodes = cypher_retriever.retrieve("What happened at Interleaf and Viaweb?")
print(nodes)

# Finally, we can use a `CypherTemplateRetriever` to provide a more constrained version of the `TextToCypherRetriever`. Rather than letting the LLM have free-range of generating any openCypher statement, we can instead provide a openCypher template and have the LLM fill in the blanks.


cypher_query = """
    MATCH (c:Chunk)-[:MENTIONS]->(o)
    WHERE o.name IN $names
    RETURN c.text, o.name, o.label;
   """


class TemplateParams(BaseModel):
    """Template params for a cypher query."""

    names: list[str] = Field(
        description="A list of entity names or keywords to use for lookup in a knowledge graph."
    )


cypher_retriever = CypherTemplateRetriever(
    index.property_graph_store, TemplateParams, cypher_query
)
nodes = cypher_retriever.retrieve("What happened at Interleaf and Viaweb?")
print(nodes)

logger.info("\n\n[DONE]", bright=True)
