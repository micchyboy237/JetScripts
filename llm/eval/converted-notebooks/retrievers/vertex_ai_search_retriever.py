from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/retrievers/vertex_ai_search_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Vertex AI Search Retriever
# 
# This notebook walks you through how to setup a Retriever that can fetch from Vertex AI search datastore

### Pre-requirements
# - Set up a Google Cloud project
# - Set up a Vertex AI Search datastore
# - Enable Vertex AI API

### Install library

# %pip install llama-index-retrievers-vertexai-search

### Restart current runtime
# 
# To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which will restart the current kernel.

import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)

### Authenticate your notebook environment (Colab only)
# 
# If you are running this notebook on Google Colab, you will need to authenticate your environment. To do this, run the new cell below. This step is not required if you are using [Vertex AI Workbench](https://cloud.google.com/vertex-ai-workbench).

import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()



from llama_index.retrievers.vertexai_search import VertexAISearchRetriever

### Set Google Cloud project information and initialize Vertex AI SDK
# 
# To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).
# 
# Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).

PROJECT_ID = "{your project id}"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)

### Test Structured datastore

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

print(retrieved_results[0])

### Test Unstructured datastore

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

print(retrieved_results2[0])

### Test Website datastore

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

print(retrieved_results3[0])

## Use in Query Engine

from llama_index.core import Settings
from llama_index.llms.vertex import Vertex
from llama_index.embeddings.vertex import VertexTextEmbedding

vertex_gemini = Vertex(
    model="gemini-1.5-pro",
    temperature=0,
    context_window=100000,
    additional_kwargs={},
)
Settings.llm = vertex_gemini

from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(struct_retriever)

response = query_engine.query("Tell me about harry potter")
print(str(response))

logger.info("\n\n[DONE]", bright=True)