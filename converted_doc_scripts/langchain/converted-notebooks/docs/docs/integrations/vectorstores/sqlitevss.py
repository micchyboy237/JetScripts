from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import SQLiteVSS
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
# SQLite-VSS

>[SQLite-VSS](https://alexgarcia.xyz/sqlite-vss/) is an `SQLite` extension designed for vector search, emphasizing local-first operations and easy integration into applications without external servers. Leveraging the `Faiss` library, it offers efficient similarity search and clustering capabilities.

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

This notebook shows how to use the `SQLiteVSS` vector database.
"""
logger.info("# SQLite-VSS")

# %pip install --upgrade --quiet  sqlite-vss

"""
## Quickstart
"""
logger.info("## Quickstart")


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
texts = [doc.page_content for doc in docs]


embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


db = SQLiteVSS.from_texts(
    texts=texts,
    embedding=embedding_function,
    table="state_union",
    db_file="/tmp/vss.db",
)

query = "What did the president say about Ketanji Brown Jackson"
data = db.similarity_search(query)

data[0].page_content

"""
## Using existing SQLite connection
"""
logger.info("## Using existing SQLite connection")


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
texts = [doc.page_content for doc in docs]


embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
connection = SQLiteVSS.create_connection(db_file="/tmp/vss.db")

db1 = SQLiteVSS(
    table="state_union", embedding=embedding_function, connection=connection
)

db1.add_texts(["Ketanji Brown Jackson is awesome"])
query = "What did the president say about Ketanji Brown Jackson"
data = db1.similarity_search(query)

data[0].page_content


os.remove("/tmp/vss.db")

logger.info("\n\n[DONE]", bright=True)