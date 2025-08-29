from google.colab import auth
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.indices.managed.vertexai import VertexAIIndex
from llama_index.llms.vertex import Vertex
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/managed/VertexAIDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Google Cloud LlamaIndex on Vertex AI for RAG

In this notebook, we will show you how to get started with the [Vertex AI RAG API](https://cloud.google.com/vertex-ai/generative-ai/docs/llamaindex-on-vertexai).

## Installation
"""
logger.info("# Google Cloud LlamaIndex on Vertex AI for RAG")

# %pip install llama-index-llms-gemini
# %pip install llama-index-indices-managed-vertexai

# %pip install llama-index
# %pip install google-cloud-aiplatform==1.53.0

"""
### Setup

Follow the steps in this documentation to create a Google Cloud project and enable the Vertex AI API.

https://cloud.google.com/vertex-ai/docs/start/cloud-environment

### Authenticating your notebook environment

* If you are using **Colab** to run this notebook, run the cell below and continue.
* If you are using **Vertex AI Workbench**, check out the setup instructions [here](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/setup-env).
"""
logger.info("### Setup")


if "google.colab" in sys.modules:

    auth.authenticate_user()

#     ! gcloud config set project {PROJECT_ID}
#     ! gcloud auth application-default login -q

"""
## Download Data
"""
logger.info("## Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
## Basic Usage

A `corpus` is a collection of `document`s. A `document` is a body of text that is broken into `chunk`s.

#### Set up LLM for RAG
"""
logger.info("## Basic Usage")


vertex_gemini = Vertex(
    model="gemini-1.5-pro-preview-0514",
    temperature=0,
    context_window=100000,
    additional_kwargs={},
)

Settings.llm = vertex_gemini


project_id = "YOUR_PROJECT_ID"
location = "us-central1"

corpus_display_name = "my-corpus"
corpus_description = "Vertex AI Corpus for LlamaIndex"

index = VertexAIIndex(
    project_id,
    location,
    corpus_display_name=corpus_display_name,
    corpus_description=corpus_description,
)
logger.debug(f"Newly created corpus name is {index.corpus_name}.")

file_name = index.insert_file(
    file_path="data/paul_graham/paul_graham_essay.txt",
    metadata={
        "display_name": "paul_graham_essay",
        "description": "Paul Graham essay",
    },
)

"""
Let's check that what we've ingested.
"""
logger.info("Let's check that what we've ingested.")

logger.debug(index.list_files())

"""
Let's ask the index a question.
"""
logger.info("Let's ask the index a question.")

query_engine = index.as_query_engine()
response = query_engine.query("What did Paul Graham do growing up?")

logger.debug(f"Response is {response.response}")

for cited_text in [node.text for node in response.source_nodes]:
    logger.debug(f"Cited text: {cited_text}")

if response.metadata:
    logger.debug(
        f"Answerability: {response.metadata.get('answerable_probability', 0)}"
    )

logger.info("\n\n[DONE]", bright=True)