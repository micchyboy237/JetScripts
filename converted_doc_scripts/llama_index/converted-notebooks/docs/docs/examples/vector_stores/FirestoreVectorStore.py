from google.colab import auth
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext, ServiceContext
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores.types import (
MetadataFilters,
ExactMatchFilter,
MetadataFilter,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.firestore import FirestoreVectorStore
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/FirestoreVectorStore.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Firestore Vector Store

# Google Firestore (Native Mode)

> [Firestore](https://cloud.google.com/firestore) is a serverless document-oriented database that scales to meet any demand. Extend your database application to build AI-powered experiences leveraging Firestore's Langchain integrations.

This notebook goes over how to use [Firestore](https://cloud.google.com/firestore) to store vectors and query them using the `FirestoreVectorStore` class.

## Before You Begin

To run this notebook, you will need to do the following:

* [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
* [Enable the Firestore API](https://console.cloud.google.com/flows/enableapi?apiid=firestore.googleapis.com)
* [Create a Firestore database](https://cloud.google.com/firestore/docs/manage-databases)

After confirmed access to database in the runtime environment of this notebook, filling the following values and run the cell before running example scripts.

## Library Installation

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙. For this notebook, we will also install `langchain-google-genai` to use Google Generative AI embeddings.
"""
logger.info("# Firestore Vector Store")

# %pip install --quiet llama-index
# %pip install --quiet llama-index-vector-stores-firestore llama-index-embeddings-huggingface

"""
### ☁ Set Your Google Cloud Project
Set your Google Cloud project so that you can leverage Google Cloud resources within this notebook.

If you don't know your project ID, try the following:

* Run `gcloud config list`.
* Run `gcloud projects list`.
* See the support page: [Locate the project ID](https://support.google.com/googleapi/answer/7014113).
"""
logger.info("### ☁ Set Your Google Cloud Project")

PROJECT_ID = "YOUR_PROJECT_ID"  # @param {type:"string"}

# !gcloud config set project {PROJECT_ID}

"""
### 🔐 Authentication

Authenticate to Google Cloud as the IAM user logged into this notebook in order to access your Google Cloud Project.

- If you are using Colab to run this notebook, use the cell below and continue.
- If you are using Vertex AI Workbench, check out the setup instructions [here](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/setup-env).
"""
logger.info("### 🔐 Authentication")


auth.authenticate_user()

"""
# Basic Usage

### Initialize FirestoreVectorStore

`FirestoreVectroStore` allows you to load data into Firestore and query it.
"""
logger.info("# Basic Usage")

COLLECTION_NAME = "test_collection"


documents = SimpleDirectoryReader(
    "../../examples/data/paul_graham"
).load_data()


embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")



store = FirestoreVectorStore(collection_name=COLLECTION_NAME)

storage_context = StorageContext.from_defaults(vector_store=store)
service_context = ServiceContext.from_defaults(
    llm=None, embed_model=embed_model
)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, service_context=service_context
)

"""
### Perform search

You can use the `FirestoreVectorStore` to perform similarity searches on the vectors you have stored. This is useful for finding similar documents or text.
"""
logger.info("### Perform search")

query_engine = index.as_query_engine()
res = query_engine.query("What did the author do growing up?")
logger.debug(str(res.source_nodes[0].text))

"""
You can apply pre-filtering to the search results by specifying a `filters` argument.
"""
logger.info("You can apply pre-filtering to the search results by specifying a `filters` argument.")


filters = MetadataFilters(
    filters=[MetadataFilter(key="author", value="Paul Graham")]
)
query_engine = index.as_query_engine(filters=filters)
res = query_engine.query("What did the author do growing up?")
logger.debug(str(res.source_nodes[0].text))

logger.info("\n\n[DONE]", bright=True)