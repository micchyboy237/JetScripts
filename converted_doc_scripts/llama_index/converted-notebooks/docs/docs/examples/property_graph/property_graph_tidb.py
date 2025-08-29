from jet.models.config import MODELS_CACHE_DIR
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import Document
from llama_index.core import PropertyGraphIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.tidb import TiDBPropertyGraphStore
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# TiDB Property Graph Index

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/property_graph/property_graph_tidb.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


TiDB is a distributed SQL database, it is MySQL compatible and features horizontal scalability, strong consistency, and high availability. Currently it only supports Vector Search in [TiDB Cloud Serverless](https://tidb.cloud/ai).

In this nodebook, we will cover how to connect to a TiDB Serverless cluster and create a property graph index.
"""
logger.info("# TiDB Property Graph Index")

# %pip install llama-index llama-index-graph-stores-tidb

"""
## Prepare TiDB Serverless Cluster

Sign up for [TiDB Cloud](https://tidb.cloud/) and create a TiDB Serverless cluster with Vector Search enabled.

Get the db connection string from the Cluster Details page, for example:

```
mysql+pymysql://user:password@host:4000/dbname?ssl_verify_cert=true&ssl_verify_identity=true
```

TiDB Serverless requires TSL connection when using public endpoint.

## Env Setup

We need just a few environment setups to get started.
"""
logger.info("## Prepare TiDB Serverless Cluster")


# os.environ["OPENAI_API_KEY"] = "sk-proj-..."

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

# import nest_asyncio

# nest_asyncio.apply()


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

"""
## Index Construction
"""
logger.info("## Index Construction")


graph_store = TiDBPropertyGraphStore(
    db_connection_string="mysql+pymysql://user:password@host:4000/dbname?ssl_verify_cert=true&ssl_verify_identity=true",
    drop_existing_table=True,
)

index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
    kg_extractors=[
        SchemaLLMPathExtractor(
            llm=OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096, temperature=0.0)
        )
    ],
    property_graph_store=graph_store,
    show_progress=True,
)

"""
## Querying and Retrieval
"""
logger.info("## Querying and Retrieval")

retriever = index.as_retriever(
    include_text=False,  # include source text in returned nodes, default True
)

nodes = retriever.retrieve("What happened at Interleaf and Viaweb?")

for node in nodes:
    logger.debug(node.text)

query_engine = index.as_query_engine(include_text=True)

response = query_engine.query("What happened at Interleaf and Viaweb?")

logger.debug(str(response))

"""
## Loading from an existing Graph

If you have an existing graph (either created with LlamaIndex or otherwise), we can connect to and use it!
"""
logger.info("## Loading from an existing Graph")


graph_store = TiDBPropertyGraphStore(
    db_connection_string="mysql+pymysql://user:password@host:4000/dbname?ssl_verify_cert=true&ssl_verify_identity=true",
)

index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    llm=OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096, temperature=0.3),
    embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
)

"""
From here, we can still insert more documents!
"""
logger.info("From here, we can still insert more documents!")


document = Document(text="LlamaIndex is great!")

index.insert(document)

nodes = index.as_retriever(include_text=False).retrieve("LlamaIndex")

logger.debug(nodes[0].text)

"""
For full details on construction, retrieval, querying of a property graph, see the [full docs page](/../../module_guides/indexing/lpg_index_guide).
"""
logger.info("For full details on construction, retrieval, querying of a property graph, see the [full docs page](/../../module_guides/indexing/lpg_index_guide).")

logger.info("\n\n[DONE]", bright=True)