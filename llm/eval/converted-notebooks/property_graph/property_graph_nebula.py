from llama_index.core import Document
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
import nest_asyncio
import os
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# NebulaGraph Property Graph Index
#
# NebulaGraph is an open-source distributed graph database built for super large-scale graphs with milliseconds of latency.
#
# If you already have an existing graph, please skip to the end of this notebook.

# %pip install llama-index llama-index-graph-stores-nebula jupyter-nebulagraph

# Docker Setup
#
# To launch NebulaGraph locally, first ensure you have docker installed. Then, you can launch the database with the following docker command.
#
# ```bash
# mkdir nebula-docker-compose
# cd nebula-docker-compose
# curl --output docker-compose.yaml https://raw.githubusercontent.com/vesoft-inc/nebula-docker-compose/master/docker-compose-lite.yaml
# docker compose up
# ```
#
# After this, you are ready to create your first property graph!
#
# > Other options/details for deploying NebulaGraph can be found in the [docs](https://docs.nebula-graph.io/):
# >
# > - [ad-hoc cluster in Google Colab](https://docs.nebula-graph.io/master/4.deployment-and-installation/2.compile-and-install-nebula-graph/8.deploy-nebula-graph-with-lite/).
# > - [Docker Desktop Extension](https://docs.nebula-graph.io/master/2.quick-start/1.quick-start-workflow/).

# %load_ext ngql
# %ngql --address 127.0.0.1 --port 9669 --user root --password nebula
# %ngql CREATE SPACE IF NOT EXISTS llamaindex_nebula_property_graph(vid_type=FIXED_STRING(256));

# %ngql USE llamaindex_nebula_property_graph;

# Env Setup
#
# We need just a few environment setups to get started.


# os.environ["OPENAI_API_KEY"] = "sk-proj-..."

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


nest_asyncio.apply()


documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/summaries/").load_data()

# We choose using gpt-4o and local embedding model intfloat/multilingual-e5-large . You can change to what you like, by editing the following lines:

# %pip install llama-index-embeddings-huggingface


Settings.llm = Ollama(model="llama3.1", request_timeout=300.0,
                      context_window=4096, temperature=0.3)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-large"
)

# Index Construction

# Prepare property graph store


graph_store = NebulaPropertyGraphStore(
    space="llamaindex_nebula_property_graph", overwrite=True
)

# And vector store:


vec_store = SimpleVectorStore()

# Finally, build the index!


index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    vector_store=vec_store,
    show_progress=True,
)

index.storage_context.vector_store.persist("./data/nebula_vec_store.json")

# Now that the graph is created, we can explore it with [jupyter-nebulagraph](https://github.com/wey-gu/jupyter_nebulagraph)

# %ngql SHOW TAGS

# %ngql SHOW EDGES

# %ngql MATCH p=(v:Entity__)-[r]->(t:Entity__) RETURN v.Entity__.name AS src, r.label AS relation, t.Entity__.name AS dest LIMIT 15;

# %ngql MATCH p=(v:Entity__)-[r]->(t:Entity__) RETURN p LIMIT 2;

# %ng_draw

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
# If you have an existing graph, we can connect to and use it!


graph_store = NebulaPropertyGraphStore(
    space="llamaindex_nebula_property_graph"
)


vec_store = SimpleVectorStore.from_persist_path("./data/nebula_vec_store.json")

index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    vector_store=vec_store,
)

# From here, we can still insert more documents!


document = Document(text="LlamaIndex is great!")

index.insert(document)

nodes = index.as_retriever(include_text=False).retrieve("LlamaIndex")

print(nodes[0].text)

logger.info("\n\n[DONE]", bright=True)
