from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from traceloop.sdk import Traceloop
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/observability/OpenLLMetry.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Observability with OpenLLMetry
[OpenLLMetry](https://github.com/traceloop/openllmetry) is an open-source project based on OpenTelemetry for tracing and monitoring
LLM applications. It connects to [all major observability platforms](https://www.traceloop.com/docs/openllmetry/integrations/introduction) (like Datadog, Dynatrace, Honeycomb, New Relic and others) and installs in minutes.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙 and OpenLLMetry.
"""
logger.info("# Observability with OpenLLMetry")

# !pip install llama-index
# !pip install traceloop-sdk

"""
## Configure API keys

Sign-up to Traceloop at [app.traceloop.com](https://app.traceloop.com). Then, go to the [API keys page](https://app.traceloop.com/settings/api-keys) and create a new API key. Copy the key and paste it in the cell below.

If you prefer to use a different observability platform like Datadog, Dynatrace, Honeycomb or others, you can find instructions on how to configure it [here](https://www.traceloop.com/docs/openllmetry/integrations/introduction).
"""
logger.info("## Configure API keys")


# os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["TRACELOOP_API_KEY"] = "..."

"""
## Initialize OpenLLMetry
"""
logger.info("## Initialize OpenLLMetry")


Traceloop.init()

"""
## Download Data
"""
logger.info("## Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


docs = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

"""
## Run a query
"""
logger.info("## Run a query")


index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
logger.debug(response)

"""
## Go to Traceloop or your favorite platform to view the results

![Traceloop](https://docs.llamaindex.ai/en/stable/_images/openllmetry.png)
"""
logger.info("## Go to Traceloop or your favorite platform to view the results")

logger.info("\n\n[DONE]", bright=True)