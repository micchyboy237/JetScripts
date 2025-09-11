from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Bagel
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
# Bagel

> [Bagel](https://www.bagel.net/) (`Open Inference platform for AI`), is like GitHub for AI data.
It is a collaborative platform where users can create,
share, and manage Inference datasets. It can support private projects for independent developers,
internal collaborations for enterprises, and public contributions for data DAOs.

### Installation and Setup

```bash
pip install bagelML langchain-community
```

## Create VectorStore from texts
"""
logger.info("# Bagel")


texts = ["hello bagel", "hello langchain", "I love salad", "my car", "a dog"]
cluster = Bagel.from_texts(cluster_name="testing", texts=texts)

cluster.similarity_search("bagel", k=3)

cluster.similarity_search_with_score("bagel", k=3)

cluster.delete_cluster()

"""
## Create VectorStore from docs
"""
logger.info("## Create VectorStore from docs")


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)[:10]

cluster = Bagel.from_documents(cluster_name="testing_with_docs", documents=docs)

query = "What did the president say about Ketanji Brown Jackson"
docs = cluster.similarity_search(query)
logger.debug(docs[0].page_content[:102])

"""
## Get all text/doc from Cluster
"""
logger.info("## Get all text/doc from Cluster")

texts = ["hello bagel", "this is langchain"]
cluster = Bagel.from_texts(cluster_name="testing", texts=texts)
cluster_data = cluster.get()

cluster_data.keys()

cluster_data

cluster.delete_cluster()

"""
## Create cluster with metadata & filter using metadata
"""
logger.info("## Create cluster with metadata & filter using metadata")

texts = ["hello bagel", "this is langchain"]
metadatas = [{"source": "notion"}, {"source": "google"}]

cluster = Bagel.from_texts(cluster_name="testing", texts=texts, metadatas=metadatas)
cluster.similarity_search_with_score("hello bagel", where={"source": "notion"})

cluster.delete_cluster()

logger.info("\n\n[DONE]", bright=True)