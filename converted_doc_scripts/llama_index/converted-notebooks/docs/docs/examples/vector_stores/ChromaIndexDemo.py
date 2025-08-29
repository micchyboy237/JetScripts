from jet.models.config import MODELS_CACHE_DIR
from IPython.display import Markdown, display
from jet.logger import CustomLogger
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import openai
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/ChromaIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Chroma

>[Chroma](https://docs.trychroma.com/getting-started) is a AI-native open-source vector database focused on developer productivity and happiness. Chroma is licensed under Apache 2.0.

<a href="https://discord.gg/MMeYNTmh3x" target="_blank">
      <img src="https://img.shields.io/discord/1073293645303795742" alt="Discord">
  </a>&nbsp;&nbsp;
  <a href="https://github.com/chroma-core/chroma/blob/master/LICENSE" target="_blank">
      <img src="https://img.shields.io/static/v1?label=license&message=Apache 2.0&color=white" alt="License">
  </a>&nbsp;&nbsp;
  <img src="https://github.com/chroma-core/chroma/actions/workflows/chroma-integration-test.yml/badge.svg?branch=main" alt="Integration Tests">

- [Website](https://www.trychroma.com/)
- [Documentation](https://docs.trychroma.com/)
- [Twitter](https://twitter.com/trychroma)
- [Discord](https://discord.gg/MMeYNTmh3x)

Chroma is fully-typed, fully-tested and fully-documented.

Install Chroma with:

```sh
pip install chromadb
```

Chroma runs in various modes. See below for examples of each integrated with LlamaIndex.
- `in-memory` - in a python script or jupyter notebook
- `in-memory with persistence` - in a script or notebook and save/load to disk
- `in a docker container` - as a server running your local machine or in the cloud

Like any other database, you can: 
- `.add` 
- `.get` 
- `.update`
- `.upsert`
- `.delete`
- `.peek`
- and `.query` runs the similarity search.

View full docs at [docs](https://docs.trychroma.com/reference).

## Basic Example

In this basic example, we take the Paul Graham essay, split it into chunks, embed it using an open-source embedding model, load it into Chroma, and then query it.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Chroma")

# %pip install llama-index-vector-stores-chroma
# %pip install llama-index-embeddings-huggingface

# !pip install llama-index

"""
#### Creating a Chroma Index
"""
logger.info("#### Creating a Chroma Index")




# import getpass

# os.environ["OPENAI_API_KEY"] = getpass.getpass("OllamaFunctionCallingAdapter API Key:")

# openai.api_key = os.environ["OPENAI_API_KEY"]

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
display(Markdown(f"<b>{response}</b>"))

"""
## Basic Example (including saving to disk)

Extending the previous example, if you want to save to disk, simply initialize the Chroma client and pass the directory where you want the data to be saved to. 

`Caution`: Chroma makes a best-effort to automatically save data to disk, however multiple in-memory clients can stomp each other's work. As a best practice, only have one client per path running at any given time.
"""
logger.info("## Basic Example (including saving to disk)")

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
display(Markdown(f"<b>{response}</b>"))

"""
## Basic Example (using the Docker Container)

You can also run the Chroma Server in a Docker container separately, create a Client to connect to it, and then pass that to LlamaIndex. 

Here is how to clone, build, and run the Docker Image:
```
git clone git@github.com:chroma-core/chroma.git
docker-compose up -d --build
```
"""
logger.info("## Basic Example (using the Docker Container)")


remote_db = chromadb.HttpClient()
chroma_collection = remote_db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
display(Markdown(f"<b>{response}</b>"))

"""
## Update and Delete

While building toward a real application, you want to go beyond adding data, and also update and delete data. 

Chroma has users provide `ids` to simplify the bookkeeping here. `ids` can be the name of the file, or a combined has like `filename_paragraphNumber`, etc.

Here is a basic example showing how to do various operations:
"""
logger.info("## Update and Delete")

doc_to_update = chroma_collection.get(limit=1)
doc_to_update["metadatas"][0] = {
    **doc_to_update["metadatas"][0],
    **{"author": "Paul Graham"},
}
chroma_collection.update(
    ids=[doc_to_update["ids"][0]], metadatas=[doc_to_update["metadatas"][0]]
)
updated_doc = chroma_collection.get(limit=1)
logger.debug(updated_doc["metadatas"][0])

logger.debug("count before", chroma_collection.count())
chroma_collection.delete(ids=[doc_to_update["ids"][0]])
logger.debug("count after", chroma_collection.count())

logger.info("\n\n[DONE]", bright=True)