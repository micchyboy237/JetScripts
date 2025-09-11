from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_text_splitters import CharacterTextSplitter
import os
import shutil
import tempfile


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
# scikit-learn

>[scikit-learn](https://scikit-learn.org/stable/) is an open-source collection of machine learning algorithms, including some implementations of the [k nearest neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). `SKLearnVectorStore` wraps this implementation and adds the possibility to persist the vector store in json, bson (binary json) or Apache Parquet format.

This notebook shows how to use the `SKLearnVectorStore` vector database.

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration
"""
logger.info("# scikit-learn")

# %pip install --upgrade --quiet  scikit-learn

# %pip install --upgrade --quiet  bson

# %pip install --upgrade --quiet  pandas pyarrow

"""
To use Ollama embeddings, you will need an Ollama key. You can get one at https://platform.ollama.com/account/api-keys or feel free to use any other embeddings.
"""
logger.info("To use Ollama embeddings, you will need an Ollama key. You can get one at https://platform.ollama.com/account/api-keys or feel free to use any other embeddings.")

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass("Enter your Ollama key:")

"""
## Basic usage

### Load a sample document corpus
"""
logger.info("## Basic usage")


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

"""
### Create the SKLearnVectorStore, index the document corpus and run a sample query
"""
logger.info(
    "### Create the SKLearnVectorStore, index the document corpus and run a sample query")


persist_path = os.path.join(tempfile.gettempdir(), "union.parquet")

vector_store = SKLearnVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_path=persist_path,  # persist_path and serializer are optional
    serializer="parquet",
)

query = "What did the president say about Ketanji Brown Jackson"
docs = vector_store.similarity_search(query)
logger.debug(docs[0].page_content)

"""
## Saving and loading a vector store
"""
logger.info("## Saving and loading a vector store")

vector_store.persist()
logger.debug("Vector store was persisted to", persist_path)

vector_store2 = SKLearnVectorStore(
    embedding=embeddings, persist_path=persist_path, serializer="parquet"
)
logger.debug("A new instance of vector store was loaded from", persist_path)

docs = vector_store2.similarity_search(query)
logger.debug(docs[0].page_content)

"""
## Clean-up
"""
logger.info("## Clean-up")

os.remove(persist_path)

logger.info("\n\n[DONE]", bright=True)
