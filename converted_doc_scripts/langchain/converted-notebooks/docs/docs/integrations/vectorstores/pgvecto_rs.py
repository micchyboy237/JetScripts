from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.fake import FakeEmbeddings
from langchain_community.vectorstores.pgvecto_rs import PGVecto_rs
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from pgvecto_rs.sdk.filters import meta_contains
from typing import List
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
# PGVecto.rs

This notebook shows how to use functionality related to the Postgres vector database ([pgvecto.rs](https://github.com/tensorchord/pgvecto.rs)).
"""
logger.info("# PGVecto.rs")

# %pip install "pgvecto_rs[sdk]" langchain-community



loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = FakeEmbeddings(size=3)

"""
Start the database with the [official demo docker image](https://github.com/tensorchord/pgvecto.rs#installation).
"""
logger.info("Start the database with the [official demo docker image](https://github.com/tensorchord/pgvecto.rs#installation).")

# ! docker run --name pgvecto-rs-demo -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d tensorchord/pgvecto-rs:latest

"""
Then contruct the db URL
"""
logger.info("Then contruct the db URL")


PORT = os.getenv("DB_PORT", 5432)
HOST = os.getenv("DB_HOST", "localhost")
USER = os.getenv("DB_USER", "postgres")
PASS = os.getenv("DB_PASS", "mysecretpassword")
DB_NAME = os.getenv("DB_NAME", "postgres")

URL = "postgresql+psycopg://{username}:{password}@{host}:{port}/{db_name}".format(
    port=PORT,
    host=HOST,
    username=USER,
    password=PASS,
    db_name=DB_NAME,
)

"""
Finally, create the VectorStore from the documents:
"""
logger.info("Finally, create the VectorStore from the documents:")

db1 = PGVecto_rs.from_documents(
    documents=docs,
    embedding=embeddings,
    db_url=URL,
    collection_name="state_of_the_union",
)

"""
You can connect to the table laterly with:
"""
logger.info("You can connect to the table laterly with:")

db1 = PGVecto_rs.from_collection_name(
    embedding=embeddings,
    db_url=URL,
    collection_name="state_of_the_union",
)

"""
Make sure that the user is permitted to create a table.

## Similarity search with score

### Similarity Search with Euclidean Distance (Default)
"""
logger.info("## Similarity search with score")

query = "What did the president say about Ketanji Brown Jackson"
docs: List[Document] = db1.similarity_search(query, k=4)
for doc in docs:
    logger.debug(doc.page_content)
    logger.debug("======================")

"""
### Similarity Search with Filter
"""
logger.info("### Similarity Search with Filter")


query = "What did the president say about Ketanji Brown Jackson"
docs: List[Document] = db1.similarity_search(
    query, k=4, filter=meta_contains({"source": "../../how_to/state_of_the_union.txt"})
)

for doc in docs:
    logger.debug(doc.page_content)
    logger.debug("======================")

"""
Or:
"""
logger.info("Or:")

query = "What did the president say about Ketanji Brown Jackson"
docs: List[Document] = db1.similarity_search(
    query, k=4, filter={"source": "../../how_to/state_of_the_union.txt"}
)

for doc in docs:
    logger.debug(doc.page_content)
    logger.debug("======================")

logger.info("\n\n[DONE]", bright=True)