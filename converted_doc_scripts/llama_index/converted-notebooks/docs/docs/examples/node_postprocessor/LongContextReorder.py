from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import LongContextReorder
from llama_index.core.response.notebook_utils import display_response
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import openai
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_postprocessor/LongContextReorder.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# LongContextReorder

Models struggle to access significant details found in the center of extended contexts. [A study](https://arxiv.org/abs/2307.03172) observed that the best performance typically arises when crucial data is positioned at the start or conclusion of the input context. Additionally, as the input context lengthens, performance drops notably, even in models designed for long contexts.

This module will re-order the retrieved nodes, which can be helpful in cases where a large top-k is needed. The reordering process works as follows:

1. Input nodes are sorted based on their relevance scores.
2. Sorted nodes are then reordered in an alternating pattern:
   - Even-indexed nodes are placed at the beginning of the new list.
   - Odd-indexed nodes are placed at the end of the new list.

This approach ensures that the highest-scored (most relevant) nodes are positioned at the beginning and end of the list, with lower-scored nodes in the middle.

## Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# LongContextReorder")

# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-llms-ollama

# !pip install llama-index


# os.environ["OPENAI_API_KEY"] = "sk-..."


Settings.llm = MLX(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.1)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()


index = VectorStoreIndex.from_documents(documents)

"""
## Run Query
"""
logger.info("## Run Query")


reorder = LongContextReorder()

reorder_engine = index.as_query_engine(
    node_postprocessors=[reorder], similarity_top_k=5
)
base_engine = index.as_query_engine(similarity_top_k=5)


base_response = base_engine.query("Did the author meet Sam Altman?")
display_response(base_response)

reorder_response = reorder_engine.query("Did the author meet Sam Altman?")
display_response(reorder_response)

"""
## Inspect Order Diffrences
"""
logger.info("## Inspect Order Diffrences")

logger.debug(base_response.get_formatted_sources())

logger.debug(reorder_response.get_formatted_sources())

logger.info("\n\n[DONE]", bright=True)