from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader, Document, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.epsilla import EpsillaVectorStore
from pyepsilla import vectordb
import logging
import openai
import os
import shutil
import sys
import textwrap


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/EpsillaIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Epsilla Vector Store
In this notebook we are going to show how to use [Epsilla](https://www.epsilla.com/) to perform vector searches in LlamaIndex.

As a prerequisite, you need to have a running Epsilla vector database (for example, through our docker image), and install the ``pyepsilla`` package.
View full docs at [docs](https://epsilla-inc.gitbook.io/epsilladb/quick-start)
"""
logger.info("# Epsilla Vector Store")

# %pip install llama-index-vector-stores-epsilla

# !pip/pip3 install pyepsilla

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.")

# !pip install llama-index




"""
### Setup MLX
Lets first begin by adding the openai api key. It will be used to created embeddings for the documents loaded into the index.
"""
logger.info("### Setup MLX")

# import getpass

# OPENAI_API_KEY = getpass.getpass("MLX API Key:")
# openai.api_key = OPENAI_API_KEY

"""
### Download Data
"""
logger.info("### Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
### Loading documents
Load documents stored in the `/data/paul_graham` folder using the SimpleDirectoryReader.
"""
logger.info("### Loading documents")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()
logger.debug(f"Total documents: {len(documents)}")
logger.debug(f"First document, id: {documents[0].doc_id}")
logger.debug(f"First document, hash: {documents[0].hash}")

"""
### Create the index
Here we create an index backed by Epsilla using the documents loaded previously. EpsillaVectorStore takes a few arguments.
- client (Any): Epsilla client to connect to.

- collection_name (str, optional): Which collection to use. Defaults to "llama_collection".
- db_path (str, optional): The path where the database will be persisted. Defaults to "/tmp/langchain-epsilla".
- db_name (str, optional): Give a name to the loaded database. Defaults to "langchain_store".
- dimension (int, optional): The dimension of the embeddings. If not provided, collection creation will be done on first insert. Defaults to None.
- overwrite (bool, optional): Whether to overwrite existing collection with same name. Defaults to False.

Epsilla vectordb is running with default host "localhost" and port "8888".
"""
logger.info("### Create the index")


client = vectordb.Client()
vector_store = EpsillaVectorStore(client=client, db_path="/tmp/llamastore")

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
### Query the data
Now we have our document stored in the index, we can ask questions against the index.
"""
logger.info("### Query the data")

query_engine = index.as_query_engine()
response = query_engine.query("Who is the author?")
logger.debug(textwrap.fill(str(response), 100))

response = query_engine.query("How did the author learn about AI?")
logger.debug(textwrap.fill(str(response), 100))

"""
Next, let's try to overwrite the previous data.
"""
logger.info("Next, let's try to overwrite the previous data.")

vector_store = EpsillaVectorStore(client=client, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
single_doc = Document(text="Epsilla is the vector database we are using.")
index = VectorStoreIndex.from_documents(
    [single_doc],
    storage_context=storage_context,
)

query_engine = index.as_query_engine()
response = query_engine.query("Who is the author?")
logger.debug(textwrap.fill(str(response), 100))

response = query_engine.query("What vector database is being used?")
logger.debug(textwrap.fill(str(response), 100))

"""
Next, let's add more data to existing collection.
"""
logger.info("Next, let's add more data to existing collection.")

vector_store = EpsillaVectorStore(client=client, overwrite=False)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
for doc in documents:
    index.insert(document=doc)

query_engine = index.as_query_engine()
response = query_engine.query("Who is the author?")
logger.debug(textwrap.fill(str(response), 100))

response = query_engine.query("What vector database is being used?")
logger.debug(textwrap.fill(str(response), 100))

logger.info("\n\n[DONE]", bright=True)