from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from llama_index.core import (
VectorStoreIndex,
SimpleDirectoryReader,
StorageContext,
load_index_from_storage,
)
from llama_index.core import Settings
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/citation_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# CitationQueryEngine

This notebook walks through how to use the CitationQueryEngine

The CitationQueryEngine can be used with any existing index.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# CitationQueryEngine")

# %pip install llama-index-embeddings-ollama
# %pip install llama-index-llms-ollama

# !pip install llama-index

"""
## Setup
"""
logger.info("## Setup")



Settings.llm = MLX(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
Settings.embed_model = MLXEmbedding(model="mxbai-embed-large")

"""
## Download Data
"""
logger.info("## Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

if not os.path.exists("./citation"):
    documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()
    index = VectorStoreIndex.from_documents(
        documents,
    )
    index.storage_context.persist(persist_dir="./citation")
else:
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./citation"),
    )

"""
## Create the CitationQueryEngine w/ Default Arguments
"""
logger.info("## Create the CitationQueryEngine w/ Default Arguments")

query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    citation_chunk_size=512,
)

response = query_engine.query("What did the author do growing up?")

logger.debug(response)

logger.debug(len(response.source_nodes))

"""
### Inspecting the Actual Source
Sources start counting at 1, but python arrays start counting at zero!

Let's confirm the source makes sense.
"""
logger.info("### Inspecting the Actual Source")

logger.debug(response.source_nodes[0].node.get_text())

logger.debug(response.source_nodes[1].node.get_text())

"""
## Adjusting Settings

Note that setting the chunk size larger than the original chunk size of the nodes will have no effect.

The default node chunk size is 1024, so here, we are not making our citation nodes any more granular.
"""
logger.info("## Adjusting Settings")

query_engine = CitationQueryEngine.from_args(
    index,
    citation_chunk_size=1024,
    similarity_top_k=3,
)

response = query_engine.query("What did the author do growing up?")

logger.debug(response)

logger.debug(len(response.source_nodes))

"""
### Inspecting the Actual Source
Sources start counting at 1, but python arrays start counting at zero!

Let's confirm the source makes sense.
"""
logger.info("### Inspecting the Actual Source")

logger.debug(response.source_nodes[0].node.get_text())

logger.info("\n\n[DONE]", bright=True)