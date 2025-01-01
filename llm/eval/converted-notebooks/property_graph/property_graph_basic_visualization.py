from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# Property Graph Index Visualization
# 
# Similar to the [property_graph_basic](property_graph_basic.ipynb) notebook, in this notebook, we demonstrate an alternative visualization approach for the default ```SimplePropertyGraphStore```
# 
# While the focus of the other notebook is querying the graph, this notebook focuses on the visualization aspect of what was created.

# %pip install llama-index

## Setup

import os
import urllib.request

# os.environ["OPENAI_API_KEY"] = "sk-proj-..."

url = "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
filename = "data/paul_graham/paul_graham_essay.txt"
os.makedirs(os.path.dirname(filename), exist_ok=True)
urllib.request.urlretrieve(url, filename)

import nest_asyncio

nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/JetScripts/llm/eval/converted-notebooks/retrievers/data/jet-resume/").load_data()

## Construction

from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

index = PropertyGraphIndex.from_documents(
    documents,
    llm=Ollama(model="llama3.2", request_timeout=300.0, context_window=4096, temperature=0.3),
    embed_model=OllamaEmbedding(model_name="nomic-embed-text"),
    show_progress=True,
)

## Visualization
# 
# Let's explore what we created. Using the ```show_jupyter_graph()``` method to create our graph directly in the Jupyter cell!
# 
# Note that this only works in Jupyter environments.

index.property_graph_store.show_jupyter_graph()

# ![example graph](./jupyter_screenshot.png)

logger.info("\n\n[DONE]", bright=True)