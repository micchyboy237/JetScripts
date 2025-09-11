from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import ManticoreSearch, ManticoreSearchSettings
from langchain_text_splitters import CharacterTextSplitter
import os
import shutil
import time


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
# ManticoreSearch VectorStore

[ManticoreSearch](https://manticoresearch.com/) is an open-source search engine that offers fast, scalable, and user-friendly capabilities. Originating as a fork of [Sphinx Search](http://sphinxsearch.com/), it has evolved to incorporate modern search engine features and improvements. ManticoreSearch distinguishes itself with its robust performance and ease of integration into various applications.

ManticoreSearch has recently introduced [vector search capabilities](https://manual.manticoresearch.com/dev/Searching/KNN), starting with search engine version 6.2 and only with [manticore-columnar-lib](https://github.com/manticoresoftware/columnar) package installed. This feature is a considerable advancement, allowing for the execution of searches based on vector similarity.

As of now, the vector search functionality is only accessible in the developmental (dev) versions of the search engine. Consequently, it is imperative to employ a developmental [manticoresearch-dev](https://pypi.org/project/manticoresearch-dev/) Python client for utilizing this feature effectively.

## Setting up environments

Starting Docker-container with ManticoreSearch and installing manticore-columnar-lib package (optional)
"""
logger.info("# ManticoreSearch VectorStore")


containers = !docker ps --filter "name=langchain-manticoresearch-server" -q
if len(containers) == 0:
#     !docker run -d -p 9308:9308 --name langchain-manticoresearch-server manticoresearch/manticore:dev
    time.sleep(20)  # Wait for the container to start up

container_id = containers[0]

# !docker exec -it --user 0 {container_id} apt-get update
# !docker exec -it --user 0 {container_id} apt-get install -y manticore-columnar-lib

# !docker restart {container_id}

"""
Installing ManticoreSearch python client
"""
logger.info("Installing ManticoreSearch python client")

# %pip install --upgrade --quiet manticoresearch-dev

"""
We want to use OllamaEmbeddings so we have to get the Ollama API Key.
"""
logger.info("We want to use OllamaEmbeddings so we have to get the Ollama API Key.")



loader = TextLoader("../../modules/paul_graham_essay.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = GPT4AllEmbeddings()

for d in docs:
    d.metadata = {"some": "metadata"}
settings = ManticoreSearchSettings(table="manticoresearch_vector_search_example")
docsearch = ManticoreSearch.from_documents(docs, embeddings, config=settings)

query = "Robert Morris is"
docs = docsearch.similarity_search(query)
logger.debug(docs)

logger.info("\n\n[DONE]", bright=True)