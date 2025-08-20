from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import set_global_handler
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import requests
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
# Logging traces with Opik

For this guide we will be downloading the essays from Paul Graham and use them as our data source. We will then start querying these essays with LlamaIndex and logging the traces to Opik.

## Creating an account on Comet.com

[Comet](https://www.comet.com/site) provides a hosted version of the Opik platform, [simply create an account](https://www.comet.com/signup?from=llm) and grab you API Key.

> You can also run the Opik platform locally, see the [installation guide](https://www.comet.com/docs/opik/self-host/self_hosting_opik/) for more information.
"""
logger.info("# Logging traces with Opik")

# import getpass

if "OPIK_API_KEY" not in os.environ:
#     os.environ["OPIK_API_KEY"] = getpass.getpass("Opik API Key: ")
if "OPIK_WORKSPACE" not in os.environ:
    os.environ["OPIK_WORKSPACE"] = input(
        "Comet workspace (often the same as your username): "
    )

"""
If you are running the Opik platform locally, simply set:
"""
logger.info("If you are running the Opik platform locally, simply set:")



"""
## Preparing our environment

First, we will install the necessary libraries, download the Chinook database and set up our different API keys.
"""
logger.info("## Preparing our environment")

# %pip install opik llama-index llama-index-agent-openai llama-index-llms-ollama --upgrade --quiet

"""
And configure the required environment variables:
"""
logger.info("And configure the required environment variables:")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass(
        "Enter your MLX API key: "
    )

"""
In addition, we will download the Paul Graham essays:
"""
logger.info("In addition, we will download the Paul Graham essays:")


os.makedirs("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/", exist_ok=True)

url = "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
response = requests.get(url)
with open("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/paul_graham_essay.txt", "wb") as f:
    f.write(response.content)

"""
## Using LlamaIndex

### Configuring the Opik integration

You can use the Opik callback directly by calling:
"""
logger.info("## Using LlamaIndex")


set_global_handler(
    "opik",
)

"""
Now that the callback handler is configured, all traces will automatically be logged to Opik.

### Using LLamaIndex

The first step is to load the data into LlamaIndex. We will use the `SimpleDirectoryReader` to load the data from the `data/paul_graham` directory. We will also create the vector store to index all the loaded documents.
"""
logger.info("### Using LLamaIndex")


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

"""
We can now query the index using the `query_engine` object:
"""
logger.info("We can now query the index using the `query_engine` object:")

response = query_engine.query("What did the author do growing up?")
logger.debug(response)

"""
You can now go to the Opik app to see the trace:

![LlamaIndex trace in Opik](https://raw.githubusercontent.com/comet-ml/opik/main/apps/opik-documentation/documentation/static/img/cookbook/llamaIndex_cookbook.png)
"""
logger.info("You can now go to the Opik app to see the trace:")

logger.info("\n\n[DONE]", bright=True)