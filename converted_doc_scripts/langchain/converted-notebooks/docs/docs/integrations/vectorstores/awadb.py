from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import AwaDB
from langchain_text_splitters import CharacterTextSplitter
import awadb
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
# AwaDB
>[AwaDB](https://github.com/awa-ai/awadb) is an AI Native database for the search and storage of embedding vectors used by LLM Applications.

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

This notebook shows how to use functionality related to the `AwaDB`.
"""
logger.info("# AwaDB")

# %pip install --upgrade --quiet  awadb


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

db = AwaDB.from_documents(docs)
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

logger.debug(docs[0].page_content)

"""
## Similarity search with score

The returned distance score is between 0-1. 0 is dissimilar, 1 is the most similar
"""
logger.info("## Similarity search with score")

docs = db.similarity_search_with_score(query)

logger.debug(docs[0])

"""
## Restore the table created and added data before

AwaDB automatically persists added document data.

If you can restore the table you created and added before, you can just do this as below:
"""
logger.info("## Restore the table created and added data before")


awadb_client = awadb.Client()
ret = awadb_client.Load("langchain_awadb")
if ret:
    logger.debug("awadb load table success")
else:
    logger.debug("awadb load table failed")

"""
awadb load table success
"""
logger.info("awadb load table success")

logger.info("\n\n[DONE]", bright=True)