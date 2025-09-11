from google.auth import compute_engine
from google.cloud import firestore
from google.cloud.firestore import Client
from google.cloud.firestore import CollectionGroup, FieldFilter, Query
from google.colab import auth
from jet.logger import logger
from langchain_core.documents import Document
from langchain_google_firestore import FirestoreLoader
from langchain_google_firestore import FirestoreSaver
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
# Google Firestore (Native Mode)

> [Firestore](https://cloud.google.com/firestore) is a serverless document-oriented database that scales to meet any demand. Extend your database application to build AI-powered experiences leveraging Firestore's Langchain integrations.

This notebook goes over how to use [Firestore](https://cloud.google.com/firestore) to [save, load and delete langchain documents](/docs/how_to#document-loaders) with `FirestoreLoader` and `FirestoreSaver`.

Learn more about the package on [GitHub](https://github.com/googleapis/langchain-google-firestore-python/).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googleapis/langchain-google-firestore-python/blob/main/docs/document_loader.ipynb)

## Before You Begin

To run this notebook, you will need to do the following:

* [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
* [Enable the Firestore API](https://console.cloud.google.com/flows/enableapi?apiid=firestore.googleapis.com)
* [Create a Firestore database](https://cloud.google.com/firestore/docs/manage-databases)

After confirmed access to database in the runtime environment of this notebook, filling the following values and run the cell before running example scripts.
"""
logger.info("# Google Firestore (Native Mode)")

SOURCE = "test"  # @param {type:"Query"|"CollectionGroup"|"DocumentReference"|"string"}

"""
### 🦜🔗 Library Installation

The integration lives in its own `langchain-google-firestore` package, so we need to install it.
"""
logger.info("### 🦜🔗 Library Installation")

# %pip install --upgrade --quiet langchain-google-firestore

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

PROJECT_ID = "my-project-id"  # @param {type:"string"}

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
## Basic Usage

### Save documents

`FirestoreSaver` can store Documents into Firestore. By default it will try to extract the Document reference from the metadata

Save langchain documents with `FirestoreSaver.upsert_documents(<documents>)`.
"""
logger.info("## Basic Usage")


saver = FirestoreSaver()

data = [Document(page_content="Hello, World!")]

saver.upsert_documents(data)

"""
#### Save documents without reference

If a collection is specified the documents will be stored with an auto generated id.
"""
logger.info("#### Save documents without reference")

saver = FirestoreSaver("Collection")

saver.upsert_documents(data)

"""
#### Save documents with other references
"""
logger.info("#### Save documents with other references")

doc_ids = ["AnotherCollection/doc_id", "foo/bar"]
saver = FirestoreSaver()

saver.upsert_documents(documents=data, document_ids=doc_ids)

"""
### Load from Collection or SubCollection

Load langchain documents with `FirestoreLoader.load()` or `Firestore.lazy_load()`. `lazy_load` returns a generator that only queries database during the iteration. To initialize `FirestoreLoader` class you need to provide:

1. `source` - An instance of a Query, CollectionGroup, DocumentReference or the single `\`-delimited path to a Firestore collection.
"""
logger.info("### Load from Collection or SubCollection")


loader_collection = FirestoreLoader("Collection")
loader_subcollection = FirestoreLoader("Collection/doc/SubCollection")


data_collection = loader_collection.load()
data_subcollection = loader_subcollection.load()

"""
### Load a single Document
"""
logger.info("### Load a single Document")


client = firestore.Client()
doc_ref = client.collection("foo").document("bar")

loader_document = FirestoreLoader(doc_ref)

data = loader_document.load()

"""
### Load from CollectionGroup or Query
"""
logger.info("### Load from CollectionGroup or Query")


col_ref = client.collection("col_group")
collection_group = CollectionGroup(col_ref)

loader_group = FirestoreLoader(collection_group)

col_ref = client.collection("collection")
query = col_ref.where(filter=FieldFilter("region", "==", "west_coast"))

loader_query = FirestoreLoader(query)

"""
### Delete documents

Delete a list of langchain documents from Firestore collection with `FirestoreSaver.delete_documents(<documents>)`.

If document ids is provided, the Documents will be ignored.
"""
logger.info("### Delete documents")

saver = FirestoreSaver()

saver.delete_documents(data)

saver.delete_documents(data, doc_ids)

"""
## Advanced Usage

### Load documents with customize document page content & metadata

The arguments of `page_content_fields` and `metadata_fields` will specify the Firestore Document fields to be written into LangChain Document `page_content` and `metadata`.
"""
logger.info("## Advanced Usage")

loader = FirestoreLoader(
    source="foo/bar/subcol",
    page_content_fields=["data_field"],
    metadata_fields=["metadata_field"],
)

data = loader.load()

"""
#### Customize Page Content Format

When the `page_content` contains only one field the information will be the field value only. Otherwise the `page_content` will be in JSON format.

### Customize Connection & Authentication
"""
logger.info("#### Customize Page Content Format")


client = Client(database="non-default-db", creds=compute_engine.Credentials())
loader = FirestoreLoader(
    source="foo",
    client=client,
)

logger.info("\n\n[DONE]", bright=True)