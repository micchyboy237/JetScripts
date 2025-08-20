from google.colab import auth
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.llms.vertex import Vertex
from llama_index.retrievers.vertexai_search import VertexAISearchRetriever
import IPython
import os
import shutil
import sys
import vertexai


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/retrievers/vertex_ai_search_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Vertex AI Search Retriever

This notebook walks you through how to setup a Retriever that can fetch from Vertex AI search datastore

### Pre-requirements
- Set up a Google Cloud project
- Set up a Vertex AI Search datastore
- Enable Vertex AI API

### Install library
"""
logger.info("# Vertex AI Search Retriever")

# %pip install llama-index-retrievers-vertexai-search

"""
### Restart current runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which will restart the current kernel.
"""
logger.info("### Restart current runtime")


app = IPython.Application.instance()
app.kernel.do_shutdown(True)

"""
### Authenticate your notebook environment (Colab only)

If you are running this notebook on Google Colab, you will need to authenticate your environment. To do this, run the new cell below. This step is not required if you are using [Vertex AI Workbench](https://cloud.google.com/vertex-ai-workbench).
"""
logger.info("### Authenticate your notebook environment (Colab only)")


if "google.colab" in sys.modules:

    auth.authenticate_user()




"""
### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).
"""
logger.info("### Set Google Cloud project information and initialize Vertex AI SDK")

PROJECT_ID = "{your project id}"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

vertexai.init(project=PROJECT_ID, location=LOCATION)

"""
### Test Structured datastore
"""
logger.info("### Test Structured datastore")

DATA_STORE_ID = "{your id}"  # @param {type:"string"}
LOCATION_ID = "global"

struct_retriever = VertexAISearchRetriever(
    project_id=PROJECT_ID,
    data_store_id=DATA_STORE_ID,
    location_id=LOCATION_ID,
    engine_data_type=1,
)

query = "harry potter"
retrieved_results = struct_retriever.retrieve(query)

logger.debug(retrieved_results[0])

"""
### Test Unstructured datastore
"""
logger.info("### Test Unstructured datastore")

DATA_STORE_ID = "{your id}"
LOCATION_ID = "global"

unstruct_retriever = VertexAISearchRetriever(
    project_id=PROJECT_ID,
    data_store_id=DATA_STORE_ID,
    location_id=LOCATION_ID,
    engine_data_type=0,
)

query = "alphabet 2018 earning"
retrieved_results2 = unstruct_retriever.retrieve(query)

logger.debug(retrieved_results2[0])

"""
### Test Website datastore
"""
logger.info("### Test Website datastore")

DATA_STORE_ID = "{your id}"
LOCATION_ID = "global"
website_retriever = VertexAISearchRetriever(
    project_id=PROJECT_ID,
    data_store_id=DATA_STORE_ID,
    location_id=LOCATION_ID,
    engine_data_type=2,
)

query = "what's diamaxol"
retrieved_results3 = website_retriever.retrieve(query)

logger.debug(retrieved_results3[0])

"""
## Use in Query Engine
"""
logger.info("## Use in Query Engine")


vertex_gemini = Vertex(
    model="gemini-1.5-pro",
    temperature=0,
    context_window=100000,
    additional_kwargs={},
)
Settings.llm = vertex_gemini


query_engine = RetrieverQueryEngine.from_args(struct_retriever)

response = query_engine.query("Tell me about harry potter")
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)