from jet.models.config import MODELS_CACHE_DIR
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import Document
from llama_index.core import PropertyGraphIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Neo4j Property Graph Index

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/property_graph/property_graph_neo4j.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


Neo4j is a production-grade graph database, capable of storing a property graph, performing vector search, filtering, and more.

The easiest way to get started is with a cloud-hosted instance using [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database/)

For this notebook, we will instead cover how to run the database locally with docker.

If you already have an existing graph, please skip to the end of this notebook.
"""
logger.info("# Neo4j Property Graph Index")

# %pip install llama-index llama-index-graph-stores-neo4j

"""
## Docker Setup

To launch Neo4j locally, first ensure you have docker installed. Then, you can launch the database with the following docker command

```bash
docker run \
    -p 7474:7474 -p 7687:7687 \
    -v $PWD/data:/data -v $PWD/plugins:/plugins \
    --name neo4j-apoc \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true \
    -e NEO4JLABS_PLUGINS=\[\"apoc\"\] \
    neo4j:latest
```

From here, you can open the db at [http://localhost:7474/](http://localhost:7474/). On this page, you will be asked to sign in. Use the default username/password of `neo4j` and `neo4j`.

Once you login for the first time, you will be asked to change the password.

After this, you are ready to create your first property graph!

## Env Setup

We need just a few environment setups to get started.
"""
logger.info("## Docker Setup")


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


graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="llamaindex",
    url="bolt://localhost:7687",
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
Now that the graph is created, we can explore it in the UI by visting [http://localhost:7474/](http://localhost:7474/). 

The easiest way to see the entire graph is to use a cypher command like `"match n=() return n"` at the top.

To delete an entire graph, a useful command is `"match n=() detach delete n"`.

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

**NOTE:** If your graph was created outside of LlamaIndex, the most useful retrievers will be [text to cypher](/../../module_guides/indexing/lpg_index_guide#texttocypherretriever) or [cypher templates](/../../module_guides/indexing/lpg_index_guide#cyphertemplateretriever). Other retrievers rely on properties that LlamaIndex inserts.
"""
logger.info("## Loading from an existing Graph")


graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="794613852",
    url="bolt://localhost:7687",
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