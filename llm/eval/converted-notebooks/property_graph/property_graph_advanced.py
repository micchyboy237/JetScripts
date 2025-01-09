from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PropertyGraphIndex
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from jet.llm.ollama.base import Ollama
from typing import Literal
import nest_asyncio
from llama_index.core import SimpleDirectoryReader
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# Property Graph Construction with Predefined Schemas
#
# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/property_graph/property_graph_advanced.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# In this notebook, we walk through using Neo4j, Ollama and Huggingface to build a property graph.
#
# Specifically, we will be using the `SchemaLLMPathExtractor` which allows us to specify an exact schema containing possible entity types, relation types, and defining how they can be connected together.
#
# This is useful for when you have a specific graph you want to build, and want to limit what the LLM is predicting.

# %pip install llama-index
# %pip install llama-index-llms-ollama
# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-graph-stores-neo4j
# %pip install llama-index-graph-stores-nebula

# Load Data

# First, lets download some sample data to play with.

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/summaries/").load_data()

# Graph Construction
#
# To construct our graph, we are going to take advantage of the `SchemaLLMPathExtractor` to construct our graph.
#
# Given some schema for a graph, we can extract entities and relations that follow this schema, rather than letting the LLM decide entities and relations at random.


nest_asyncio.apply()


entities = Literal["PERSON", "PLACE", "ORGANIZATION"]
relations = Literal["HAS", "PART_OF", "WORKED_ON", "WORKED_WITH", "WORKED_AT"]

validation_schema = {
    "PERSON": ["HAS", "PART_OF", "WORKED_ON", "WORKED_WITH", "WORKED_AT"],
    "PLACE": ["HAS", "PART_OF", "WORKED_AT"],
    "ORGANIZATION": ["HAS", "PART_OF", "WORKED_WITH"],
}
validation_schema = [
    ("ORGANIZATION", "HAS", "PERSON"),
    ("PERSON", "WORKED_AT", "ORGANIZATION"),
    ("PERSON", "WORKED_WITH", "PERSON"),
    ("PERSON", "WORKED_ON", "ORGANIZATION"),
    ("PERSON", "PART_OF", "ORGANIZATION"),
    ("ORGANIZATION", "PART_OF", "ORGANIZATION"),
    ("PERSON", "WORKED_AT", "PLACE"),
]

kg_extractor = SchemaLLMPathExtractor(
    llm=Ollama(model="llama3", json_mode=True, request_timeout=3600),
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    strict=True,
)

# Now, You can use SimplePropertyGraph, Neo4j, or NebulaGraph to store the graph.
#
# **Option 1. Neo4j**
#
# To launch Neo4j locally, first ensure you have docker installed. Then, you can launch the database with the following docker command
#
# ```bash
# docker run \
#     -p 7474:7474 -p 7687:7687 \
#     -v $PWD/data:/data -v $PWD/plugins:/plugins \
#     --name neo4j-apoc \
#     -e NEO4J_apoc_export_file_enabled=true \
#     -e NEO4J_apoc_import_file_enabled=true \
#     -e NEO4J_apoc_import_file_use__neo4j__config=true \
#     -e NEO4JLABS_PLUGINS=\[\"apoc\"\] \
#     neo4j:latest
# ```
#
# From here, you can open the db at [http://localhost:7474/](http://localhost:7474/). On this page, you will be asked to sign in. Use the default username/password of `neo4j` and `neo4j`.
#
# Once you login for the first time, you will be asked to change the password.
#
# After this, you are ready to create your first property graph!


graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="<password>",
    url="bolt://localhost:7687",
)
vec_store = None

# **Option 2. NebulaGraph**
#
# To launch NebulaGraph locally, first ensure you have docker installed. Then, you can launch the database with the following docker command.
#
# ```bash
# mkdir nebula-docker-compose
# cd nebula-docker-compose
# curl --output docker-compose.yaml https://raw.githubusercontent.com/vesoft-inc/nebula-docker-compose/master/docker-compose-lite.yaml
# docker compose up
# ```
# After this, you are ready to create your first property graph!
#
# > Other options/details for deploying NebulaGraph can be found in the [docs](https://docs.nebula-graph.io/):
# >
# > - [ad-hoc cluster in Google Colab](https://docs.nebula-graph.io/master/4.deployment-and-installation/2.compile-and-install-nebula-graph/8.deploy-nebula-graph-with-lite/).
# > - [Docker Desktop Extension](https://docs.nebula-graph.io/master/2.quick-start/1.quick-start-workflow/).


graph_store = NebulaPropertyGraphStore(
    space="llamaindex_nebula_property_graph", overwrite=True
)
vec_store = SimpleVectorStore()

# *If you want to explore the graph with NebulaGraph Jupyter extension*, run the following commands. Or just skip them.

# %pip install jupyter-nebulagraph

# %load_ext ngql
# %ngql --address 127.0.0.1 --port 9669 --user root --password nebula
# %ngql CREATE SPACE IF NOT EXISTS llamaindex_nebula_property_graph(vid_type=FIXED_STRING(256));

# %ngql USE llamaindex_nebula_property_graph;

# **Start building!**
#
# **NOTE:** Using a local model will be slower when extracting compared to API based models. Local models (like Ollama) are typically limited to sequential processing. Expect this to take about 10 minutes on an M2 Max.


index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[kg_extractor],
    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    property_graph_store=graph_store,
    vector_store=vec_store,
    show_progress=True,
)

# If we inspect the graph created, we can see that it only includes the relations and entity types that we defined!

# %ngql MATCH p=()-[]->() RETURN p LIMIT 20;

# %ng_draw

# Or Neo4j:

# ![local graph](./local_kg.png)

# For information on all `kg_extractors`, see [the documentation](/../../module_guides/indexing/lpg_index_guide#construction).

# Querying
#
# Now that our graph is created, we can query it.
#
# As is the theme with this notebook, we will be using a lower-level API and constructing all our retrievers ourselves!


llm_synonym = LLMSynonymRetriever(
    index.property_graph_store,
    llm=Ollama(model="llama3", request_timeout=3600),
    include_text=False,
)
vector_context = VectorContextRetriever(
    index.property_graph_store,
    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    include_text=False,
)

retriever = index.as_retriever(
    sub_retrievers=[
        llm_synonym,
        vector_context,
    ]
)

nodes = retriever.retrieve("What happened at Interleaf?")

for node in nodes:
    print(node.text)

# We can also create a query engine with similar syntax.

query_engine = index.as_query_engine(
    sub_retrievers=[
        llm_synonym,
        vector_context,
    ],
    llm=Ollama(model="llama3", request_timeout=3600),
)

response = query_engine.query("What happened at Interleaf?")

print(str(response))

# For more info on all retrievers, see the [complete guide](/../../module_guides/indexing/lpg_index_guide#retrieval-and-querying).

logger.info("\n\n[DONE]", bright=True)
