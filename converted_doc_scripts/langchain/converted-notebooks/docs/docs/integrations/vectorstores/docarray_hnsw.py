from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import DocArrayHnswSearch
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
# DocArray HnswSearch

>[DocArrayHnswSearch](https://docs.docarray.org/user_guide/storing/index_hnswlib/) is a lightweight Document Index implementation provided by [Docarray](https://github.com/docarray/docarray) that runs fully locally and is best suited for small- to medium-sized datasets. It stores vectors on disk in [hnswlib](https://github.com/nmslib/hnswlib), and stores all other data in [SQLite](https://www.sqlite.org/index.html).

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

This notebook shows how to use functionality related to the `DocArrayHnswSearch`.

## Setup

Uncomment the below cells to install docarray and get/set your Ollama api key if you haven't already done so.
"""
logger.info("# DocArray HnswSearch")

# %pip install --upgrade --quiet  "docarray[hnswlib]"


"""
## Using DocArrayHnswSearch
"""
logger.info("## Using DocArrayHnswSearch")


documents = TextLoader("../../how_to/state_of_the_union.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = DocArrayHnswSearch.from_documents(
    docs, embeddings, work_dir="hnswlib_store/", n_dim=1536
)

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


shutil.rmtree("hnswlib_store")

logger.info("\n\n[DONE]", bright=True)
