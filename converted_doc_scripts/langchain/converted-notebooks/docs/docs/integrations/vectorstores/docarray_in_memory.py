from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
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
# DocArray InMemorySearch

>[DocArrayInMemorySearch](https://docs.docarray.org/user_guide/storing/index_in_memory/) is a document index provided by [Docarray](https://github.com/docarray/docarray) that stores documents in memory. It is a great starting point for small datasets, where you may not want to launch a database server.

This notebook shows how to use functionality related to the `DocArrayInMemorySearch`.

## Setup

Uncomment the below cells to install docarray and get/set your Ollama api key if you haven't already done so.
"""
logger.info("# DocArray InMemorySearch")

# %pip install --upgrade --quiet  langchain-community "docarray"


"""
## Using DocArrayInMemorySearch
"""
logger.info("## Using DocArrayInMemorySearch")


documents = TextLoader("../../how_to/state_of_the_union.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db = DocArrayInMemorySearch.from_documents(docs, embeddings)

"""
### Similarity search
"""
logger.info("### Similarity search")

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

logger.debug(docs[0].page_content)

"""
### Similarity search with score

The returned distance score is cosine distance. Therefore, a lower score is better.
"""
logger.info("### Similarity search with score")

docs = db.similarity_search_with_score(query)

docs[0]

logger.info("\n\n[DONE]", bright=True)
