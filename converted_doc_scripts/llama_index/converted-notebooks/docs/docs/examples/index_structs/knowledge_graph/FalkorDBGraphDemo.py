from IPython.display import Markdown, display
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex
from llama_index.core import StorageContext
from llama_index.graph_stores.falkordb import FalkorDBGraphStore
from pyvis.network import Network
import logging
import os
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/index_structs/knowledge_graph/FalkorDBGraphDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# FalkorDB Graph Store

This notebook walks through configuring `FalkorDB` to be the backend for graph storage in LlamaIndex.
"""
logger.info("# FalkorDB Graph Store")

# %pip install llama-index-llms-ollama
# %pip install llama-index-graph-stores-falkordb


# os.environ["OPENAI_API_KEY"] = "API_KEY_HERE"


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

"""
## Using Knowledge Graph with FalkorDBGraphStore

### Start FalkorDB

The easiest way to start FalkorDB as a Graph database is using the [falkordb](https://hub.docker.com/r/falkordb/falkordb:edge) docker image.

To follow every step of this tutorial, launch the image as follows:

```bash
docker run -p 6379:6379 -it --rm falkordb/falkordb:edge
```
"""
logger.info("## Using Knowledge Graph with FalkorDBGraphStore")


graph_store = FalkorDBGraphStore(
    "redis://localhost:6379", decode_responses=True
)

"""
#### Building the Knowledge Graph
"""
logger.info("#### Building the Knowledge Graph")


documents = SimpleDirectoryReader(
    "../../../../examples/paul_graham_essay/data"
).load_data()

llm = OllamaFunctionCallingAdapter(temperature=0, model="llama3.2")
Settings.llm = llm
Settings.chunk_size = 512


storage_context = StorageContext.from_defaults(graph_store=graph_store)

index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    storage_context=storage_context,
)

"""
#### Querying the Knowledge Graph

First, we can query and send only the triplets to the LLM.
"""
logger.info("#### Querying the Knowledge Graph")

query_engine = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)
response = query_engine.query(
    "Tell me more about Interleaf",
)

display(Markdown(f"<b>{response}</b>"))

"""
For more detailed answers, we can also send the text from where the retrieved tripets were extracted.
"""
logger.info("For more detailed answers, we can also send the text from where the retrieved tripets were extracted.")

query_engine = index.as_query_engine(
    include_text=True, response_mode="tree_summarize"
)
response = query_engine.query(
    "Tell me more about Interleaf",
)

display(Markdown(f"<b>{response}</b>"))

"""
#### Visualizing the Graph
"""
logger.info("#### Visualizing the Graph")

# %pip install pyvis


g = index.get_networkx_graph()
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
net.show("falkordbgraph_draw.html")

logger.info("\n\n[DONE]", bright=True)
