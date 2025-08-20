from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import PropertyGraphIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import shutil
import urllib.request


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Property Graph Index Visualization

Similar to the [property_graph_basic](property_graph_basic.ipynb) notebook, in this notebook, we demonstrate an alternative visualization approach for the default ```SimplePropertyGraphStore```

While the focus of the other notebook is querying the graph, this notebook focuses on the visualization aspect of what was created.
"""
logger.info("# Property Graph Index Visualization")

# %pip install llama-index

"""
## Setup
"""
logger.info("## Setup")


# os.environ["OPENAI_API_KEY"] = "sk-proj-..."

url = "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
filename = f"{GENERATED_DIR}/paul_graham/paul_graham_essay.txt"
os.makedirs(os.path.dirname(filename), exist_ok=True)
urllib.request.urlretrieve(url, filename)

# import nest_asyncio

# nest_asyncio.apply()


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

"""
## Construction
"""
logger.info("## Construction")


index = PropertyGraphIndex.from_documents(
    documents,
    llm=MLX(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.3),
    embed_model=MLXEmbedding(model_name="mxbai-embed-large"),
    show_progress=True,
)

"""
## Visualization

Let's explore what we created. Using the ```show_jupyter_graph()``` method to create our graph directly in the Jupyter cell!

Note that this only works in Jupyter environments.
"""
logger.info("## Visualization")

index.property_graph_store.show_jupyter_graph()

"""
![example graph](./jupyter_screenshot.png)
"""

logger.info("\n\n[DONE]", bright=True)