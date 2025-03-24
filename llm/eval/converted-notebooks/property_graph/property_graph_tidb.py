from llama_index.core import Document
from llama_index.graph_stores.tidb import TiDBPropertyGraphStore
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from jet.llm.ollama.base import Ollama
from jet.llm.ollama.base import OllamaEmbedding
from llama_index.core import PropertyGraphIndex
from llama_index.core import SimpleDirectoryReader
import nest_asyncio
import os
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# TiDB Property Graph Index
#
# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/property_graph/property_graph_tidb.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
#
# TiDB is a distributed SQL database, it is MySQL compatible and features horizontal scalability, strong consistency, and high availability. Currently it only supports Vector Search in [TiDB Cloud Serverless](https://tidb.cloud/ai).
#
# In this nodebook, we will cover how to connect to a TiDB Serverless cluster and create a property graph index.

# %pip install llama-index llama-index-graph-stores-tidb

# Prepare TiDB Serverless Cluster
#
# Sign up for [TiDB Cloud](https://tidb.cloud/) and create a TiDB Serverless cluster with Vector Search enabled.
#
# Get the db connection string from the Cluster Details page, for example:
#
# ```
# mysql+pymysql://user:password@host:4000/dbname?ssl_verify_cert=true&ssl_verify_identity=true
# ```
#
# TiDB Serverless requires TSL connection when using public endpoint.

# Env Setup
#
# We need just a few environment setups to get started.


# os.environ["OPENAI_API_KEY"] = "sk-proj-..."

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


nest_asyncio.apply()


documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

# Index Construction


graph_store = TiDBPropertyGraphStore(
    db_connection_string="mysql+pymysql://user:password@host:4000/dbname?ssl_verify_cert=true&ssl_verify_identity=true",
    drop_existing_table=True,
)

index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=OllamaEmbedding(model_name="nomic-embed-text"),
    kg_extractors=[
        SchemaLLMPathExtractor(
            llm=Ollama(model="llama3.2", request_timeout=300.0,
                       context_window=4096, temperature=0.0)
        )
    ],
    property_graph_store=graph_store,
    show_progress=True,
)

# Querying and Retrieval

retriever = index.as_retriever(
    include_text=False,  # include source text in returned nodes, default True
)

nodes = retriever.retrieve("What happened at Interleaf and Viaweb?")

for node in nodes:
    print(node.text)

query_engine = index.as_query_engine(include_text=True)

response = query_engine.query("What happened at Interleaf and Viaweb?")

print(str(response))

# Loading from an existing Graph
#
# If you have an existing graph (either created with LlamaIndex or otherwise), we can connect to and use it!


graph_store = TiDBPropertyGraphStore(
    db_connection_string="mysql+pymysql://user:password@host:4000/dbname?ssl_verify_cert=true&ssl_verify_identity=true",
)

index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    llm=Ollama(model="llama3.2", request_timeout=300.0,
               context_window=4096, temperature=0.3),
    embed_model=OllamaEmbedding(model_name="nomic-embed-text"),
)

# From here, we can still insert more documents!


document = Document(text="LlamaIndex is great!")

index.insert(document)

nodes = index.as_retriever(include_text=False).retrieve("LlamaIndex")

print(nodes[0].text)

# For full details on construction, retrieval, querying of a property graph, see the [full docs page](/../../module_guides/indexing/lpg_index_guide).

logger.info("\n\n[DONE]", bright=True)
