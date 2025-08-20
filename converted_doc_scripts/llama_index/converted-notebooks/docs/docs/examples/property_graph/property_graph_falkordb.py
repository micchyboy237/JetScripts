from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Document
from llama_index.core import PropertyGraphIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.falkordb import FalkorDBPropertyGraphStore
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
# FalkorDB Property Graph Index

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/property_graph/property_graph_falkordb.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


FalkorDB is a production-grade graph database, capable of storing a property graph, performing vector search, filtering, and more.

The easiest way to get started is with a cloud-hosted instance using [FalkorDB Cloud](https://app.falkordb.cloud)

For this notebook, we will instead cover how to run the database locally with docker.

If you already have an existing graph, please skip to the end of this notebook.
"""
logger.info("# FalkorDB Property Graph Index")

# %pip install llama-index llama-index-graph-stores-falkordb

"""
## Docker Setup

To launch FalkorDB locally, first ensure you have docker installed. Then, you can launch the database with the following docker command

```bash
docker run \
    -p 3000:3000 -p 6379:6379 \
    -v $PWD/data:/data \
    falkordb/falkordb:latest
```

From here, you can open the db at [http://localhost:3000/](http://localhost:3000/). On this page, you will be asked to sign in.

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


graph_store = FalkorDBPropertyGraphStore(
    url="falkor://localhost:6379",
)


index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=MLXEmbedding(model_name="mxbai-embed-large"),
    kg_extractors=[
        SchemaLLMPathExtractor(
            llm=MLXLlamaIndexLLMAdapter(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.0)
        )
    ],
    property_graph_store=graph_store,
    show_progress=True,
)

"""
Now that the graph is created, we can explore it in the UI by visting [http://localhost:3000/](http://localhost:3000/). 

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


graph_store = FalkorDBPropertyGraphStore(
    url="falkor://localhost:6379",
)

index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    llm=MLXLlamaIndexLLMAdapter(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.3),
    embed_model=MLXEmbedding(model_name="mxbai-embed-large"),
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