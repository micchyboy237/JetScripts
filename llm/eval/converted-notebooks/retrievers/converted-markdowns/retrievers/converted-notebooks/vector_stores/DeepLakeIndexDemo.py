from jet.llm.ollama.base import OllamaEmbedding
from jet.llm.ollama.constants import OLLAMA_SMALL_EMBED_MODEL
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
import os
import textwrap
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
import urllib.request
from llama_index.core import StorageContext
import deeplake

initialize_ollama_settings()

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/DeepLakeIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# Deep Lake Vector Store Quickstart
"""

"""
Deep Lake can be installed using pip.
"""

# %pip install llama-index-vector-stores-deeplake

# !pip install llama-index
# !pip install deeplake

"""
Next, let's import the required modules and set the needed environmental variables:
"""


# os.environ["OPENAI_API_KEY"] = "sk-********************************"
# os.environ["ACTIVELOOP_TOKEN"] = "********************************"

"""
We are going to embed and store one of Paul Graham's essays in a Deep Lake Vector Store stored locally. First, we download the data to a directory called `data/paul_graham`
"""


# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt",
#     f"{GENERATED_DIR}/paul_graham/paul_graham_essay.txt",
# )

"""
We can now create documents from the source data file.
"""

DATA_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
documents = SimpleDirectoryReader(DATA_DIR).load_data()
print(
    "Document ID:",
    documents[0].doc_id,
    "Document Hash:",
    documents[0].hash,
)

"""
Finally, let's create the Deep Lake Vector Store and populate it with data. We use a default tensor configuration, which creates tensors with `text (str)`, `metadata(json)`, `id (str, auto-populated)`, `embedding (float32)`. [Learn more about tensor customizability here](https://docs.activeloop.ai/example-code/getting-started/vector-store/step-4-customizing-vector-stores).
"""

VECTOR_STORE_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/graphs/generated/run_deeplake/pg_essay_deeplake"


def ollama_embedding_function(texts, model=OLLAMA_SMALL_EMBED_MODEL):
    if isinstance(texts, str):
        texts = [texts]

    embed_model = OllamaEmbedding(model_name=model)
    results = embed_model.get_general_text_embedding(texts)
    return results


EMBEDDING_FUNCTION = ollama_embedding_function

vector_store = DeepLakeVectorStore(
    dataset_path=VECTOR_STORE_PATH,
    overwrite=False,
    embedding_function=EMBEDDING_FUNCTION,
    read_only=False,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
## Performing Vector Search

Deep Lake offers highly-flexible vector search and hybrid search options [discussed in detail in these tutorials](https://docs.activeloop.ai/example-code/tutorials/vector-store/vector-search-options). In this Quickstart, we show a simple example using default options.
"""

query_engine = index.as_query_engine()
response = query_engine.query(
    "Who is Jethro?",
)

print(textwrap.fill(str(response), 100))

response = query_engine.query("What was a hard moment for the author?")

print(textwrap.fill(str(response), 100))

query_engine = index.as_query_engine()
response = query_engine.query("What was a hard moment for the author?")
print(textwrap.fill(str(response), 100))

"""
## Deleting items from the database
"""

"""
To find the id of a document to delete, you can query the underlying deeplake dataset directly
"""


ds = deeplake.load(VECTOR_STORE_PATH)

idx = ds.id[0].numpy().tolist()
index.delete(idx[0])
logger.log("Deleted idx:", idx, colors=["DEBUG", "SUCCESS"])

logger.info("\n\n[DONE]", bright=True)
