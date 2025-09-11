from dotenv import load_dotenv
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import (
Kinetica,
KineticaSettings,
)
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
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
# Kinetica Vectorstore based Retriever

>[Kinetica](https://www.kinetica.com/) is a database with integrated support for vector similarity search

It supports:
- exact and approximate nearest neighbor search
- L2 distance, inner product, and cosine distance

This notebook shows how to use a retriever based on Kinetica vector store (`Kinetica`).
"""
logger.info("# Kinetica Vectorstore based Retriever")

# %pip install gpudb==7.2.0.9

"""
We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.
"""
logger.info("We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")


load_dotenv()


HOST = os.getenv("KINETICA_HOST", "http://127.0.0.1:9191")
USERNAME = os.getenv("KINETICA_USERNAME", "")
PASSWORD = os.getenv("KINETICA_PASSWORD", "")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def create_config() -> KineticaSettings:
    return KineticaSettings(host=HOST, username=USERNAME, password=PASSWORD)

"""
## Create Retriever from vector store
"""
logger.info("## Create Retriever from vector store")

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")


COLLECTION_NAME = "state_of_the_union_test"
connection = create_config()

db = Kinetica.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    config=connection,
)

retriever = db.as_retriever(search_kwargs={"k": 2})

"""
## Search with retriever
"""
logger.info("## Search with retriever")

result = retriever.get_relevant_documents(
    "What did the president say about Ketanji Brown Jackson"
)
logger.debug(docs[0].page_content)

logger.info("\n\n[DONE]", bright=True)