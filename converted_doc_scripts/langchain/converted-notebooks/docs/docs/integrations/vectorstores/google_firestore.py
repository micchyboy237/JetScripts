from google.api_core.client_options import ClientOptions
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from google.colab import auth
from jet.logger import logger
from langchain_core.documents import Document
from langchain_google_firestore import FirestoreVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
---
sidebar_label: Firestore
---

# Google Firestore (Native Mode)

> [Firestore](https://cloud.google.com/firestore) is a serverless document-oriented database that scales to meet any demand. Extend your database application to build AI-powered experiences leveraging Firestore's Langchain integrations.

This notebook goes over how to use [Firestore](https://cloud.google.com/firestore) to store vectors and query them using the `FirestoreVectorStore` class.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googleapis/langchain-google-firestore-python/blob/main/docs/vectorstores.ipynb)

## Before You Begin

To run this notebook, you will need to do the following:

* [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
* [Enable the Firestore API](https://console.cloud.google.com/flows/enableapi?apiid=firestore.googleapis.com)
* [Create a Firestore database](https://cloud.google.com/firestore/docs/manage-databases)

After confirmed access to database in the runtime environment of this notebook, filling the following values and run the cell before running example scripts.
"""
logger.info("# Google Firestore (Native Mode)")

COLLECTION_NAME = "test"  # @param {type:"CollectionReference"|"string"}

"""
### 🦜🔗 Library Installation

The integration lives in its own `langchain-google-firestore` package, so we need to install it. For this notebook, we will also install `langchain-google-genai` to use Google Generative AI embeddings.
"""
logger.info("### 🦜🔗 Library Installation")

# %pip install -upgrade --quiet langchain-google-firestore langchain-google-vertexai

"""
**Colab only**: Uncomment the following cell to restart the kernel or use the button to restart the kernel. For Vertex AI Workbench you can restart the terminal using the button on top.
"""



"""
### ☁ Set Your Google Cloud Project
Set your Google Cloud project so that you can leverage Google Cloud resources within this notebook.

If you don't know your project ID, try the following:

* Run `gcloud config list`.
* Run `gcloud projects list`.
* See the support page: [Locate the project ID](https://support.google.com/googleapi/answer/7014113).
"""
logger.info("### ☁ Set Your Google Cloud Project")

PROJECT_ID = "extensions-testing"  # @param {type:"string"}

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

`FirestoreVectorStore` allows you to store new vectors in a Firestore database. You can use it to store embeddings from any model, including those from Google Generative AI.
"""
logger.info("# Basic Usage")


embedding = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest",
    project=PROJECT_ID,
)

ids = ["apple", "banana", "orange"]
fruits_texts = ['{"name": "apple"}', '{"name": "banana"}', '{"name": "orange"}']

vector_store = FirestoreVectorStore(
    collection="fruits",
    embedding=embedding,
)

vector_store.add_texts(fruits_texts, ids=ids)

"""
As a shorthand, you can initilize and add vectors in a single step using the `from_texts` and `from_documents` method.
"""
logger.info("As a shorthand, you can initilize and add vectors in a single step using the `from_texts` and `from_documents` method.")

vector_store = FirestoreVectorStore.from_texts(
    collection="fruits",
    texts=fruits_texts,
    embedding=embedding,
)


fruits_docs = [Document(page_content=fruit) for fruit in fruits_texts]

vector_store = FirestoreVectorStore.from_documents(
    collection="fruits",
    documents=fruits_docs,
    embedding=embedding,
)

"""
### Delete Vectors

You can delete documents with vectors from the database using the `delete` method. You'll need to provide the document ID of the vector you want to delete. This will remove the whole document from the database, including any other fields it may have.
"""
logger.info("### Delete Vectors")

vector_store.delete(ids)

"""
### Update Vectors

Updating vectors is similar to adding them. You can use the `add` method to update the vector of a document by providing the document ID and the new vector.
"""
logger.info("### Update Vectors")

fruit_to_update = ['{"name": "apple","price": 12}']
apple_id = "apple"

vector_store.add_texts(fruit_to_update, ids=[apple_id])

"""
## Similarity Search

You can use the `FirestoreVectorStore` to perform similarity searches on the vectors you have stored. This is useful for finding similar documents or text.
"""
logger.info("## Similarity Search")

vector_store.similarity_search("I like fuji apples", k=3)

vector_store.max_marginal_relevance_search("fuji", 5)

"""
You can add a pre-filter to the search by using the `filters` parameter. This is useful for filtering by a specific field or value.
"""
logger.info("You can add a pre-filter to the search by using the `filters` parameter. This is useful for filtering by a specific field or value.")


vector_store.max_marginal_relevance_search(
    "fuji", 5, filters=FieldFilter("content", "==", "apple")
)

"""
### Customize Connection & Authentication
"""
logger.info("### Customize Connection & Authentication")


client_options = ClientOptions()
client = firestore.Client(client_options=client_options)

vector_store = FirestoreVectorStore(
    collection="fruits",
    embedding=embedding,
    client=client,
)

logger.info("\n\n[DONE]", bright=True)