from jet.logger import CustomLogger
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
import deeplake
import os
import shutil
import textwrap
import urllib.request


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/DeepLakeIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Deep Lake Vector Store Quickstart

Deep Lake can be installed using pip.
"""
logger.info("# Deep Lake Vector Store Quickstart")

# %pip install llama-index-vector-stores-deeplake

# !pip install llama-index
# !pip install deeplake

"""
Next, let's import the required modules and set the needed environmental variables:
"""
logger.info(
    "Next, let's import the required modules and set the needed environmental variables:")


# os.environ["OPENAI_API_KEY"] = "sk-********************************"
os.environ["ACTIVELOOP_TOKEN"] = "********************************"

"""
We are going to embed and store one of Paul Graham's essays in a Deep Lake Vector Store stored locally. First, we download the data to a directory called `data/paul_graham`
"""
logger.info("We are going to embed and store one of Paul Graham's essays in a Deep Lake Vector Store stored locally. First, we download the data to a directory called `data/paul_graham`")


urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt",
    "data/paul_graham/paul_graham_essay.txt",
)

"""
We can now create documents from the source data file.
"""
logger.info("We can now create documents from the source data file.")

documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()
logger.debug(
    "Document ID:",
    documents[0].doc_id,
    "Document Hash:",
    documents[0].hash,
)

"""
Finally, let's create the Deep Lake Vector Store and populate it with data. We use a default tensor configuration, which creates tensors with `text (str)`, `metadata(json)`, `id (str, auto-populated)`, `embedding (float32)`. [Learn more about tensor customizability here](https://docs.activeloop.ai/example-code/getting-started/vector-store/step-4-customizing-vector-stores).
"""
logger.info(
    "Finally, let's create the Deep Lake Vector Store and populate it with data. We use a default tensor configuration, which creates tensors with `text (str)`, `metadata(json)`, `id (str, auto-populated)`, `embedding (float32)`. [Learn more about tensor customizability here](https://docs.activeloop.ai/example-code/getting-started/vector-store/step-4-customizing-vector-stores).")


dataset_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/tempset/paul_graham"

vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
## Performing Vector Search

Deep Lake offers highly-flexible vector search and hybrid search options [discussed in detail in these tutorials](https://docs.activeloop.ai/example-code/tutorials/vector-store/vector-search-options). In this Quickstart, we show a simple example using default options.
"""
logger.info("## Performing Vector Search")

query_engine = index.as_query_engine()
response = query_engine.query(
    "What did the author learn?",
)

logger.debug(textwrap.fill(str(response), 100))

response = query_engine.query("What was a hard moment for the author?")

logger.debug(textwrap.fill(str(response), 100))

query_engine = index.as_query_engine()
response = query_engine.query("What was a hard moment for the author?")
logger.debug(textwrap.fill(str(response), 100))

"""
## Deleting items from the database

To find the id of a document to delete, you can query the underlying deeplake dataset directly
"""
logger.info("## Deleting items from the database")


ds = deeplake.load(dataset_path)

idx = ds.id[0].numpy().tolist()
idx

index.delete(idx[0])

logger.info("\n\n[DONE]", bright=True)
