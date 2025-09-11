from jet.models.config import MODELS_CACHE_DIR
from annoy import AnnoyIndex
from jet.logger import logger
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Annoy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import os
import shutil
import uuid


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
# Annoy

> [Annoy](https://github.com/spotify/annoy) (`Approximate Nearest Neighbors Oh Yeah`) is a C++ library with Python bindings to search for points in space that are close to a given query point. It also creates large read-only file-based data structures that are mapped into memory so that many processes may share the same data.

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

This notebook shows how to use functionality related to the `Annoy` vector database.

```{note}
NOTE: Annoy is read-only - once the index is built you cannot add any more embeddings!
If you want to progressively add new entries to your VectorStore then better choose an alternative!
```
"""
logger.info("# Annoy")

# %pip install --upgrade --quiet  annoy

"""
## Create VectorStore from texts
"""
logger.info("## Create VectorStore from texts")


model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings_func = HuggingFaceEmbeddings(model_name=model_name)

texts = ["pizza is great", "I love salad", "my car", "a dog"]

vector_store = Annoy.from_texts(texts, embeddings_func)

vector_store_v2 = Annoy.from_texts(
    texts, embeddings_func, metric="dot", n_trees=100, n_jobs=1
)

vector_store.similarity_search("food", k=3)

vector_store.similarity_search_with_score("food", k=3)

"""
## Create VectorStore from docs
"""
logger.info("## Create VectorStore from docs")


loader = TextLoader("../../how_to/state_of_the_union.txtn.txtn.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

docs[:5]

vector_store_from_docs = Annoy.from_documents(docs, embeddings_func)

query = "What did the president say about Ketanji Brown Jackson"
docs = vector_store_from_docs.similarity_search(query)

logger.debug(docs[0].page_content[:100])

"""
## Create VectorStore via existing embeddings
"""
logger.info("## Create VectorStore via existing embeddings")

embs = embeddings_func.embed_documents(texts)

data = list(zip(texts, embs))

vector_store_from_embeddings = Annoy.from_embeddings(data, embeddings_func)

vector_store_from_embeddings.similarity_search_with_score("food", k=3)

"""
## Search via embeddings
"""
logger.info("## Search via embeddings")

motorbike_emb = embeddings_func.embed_query("motorbike")

vector_store.similarity_search_by_vector(motorbike_emb, k=3)

vector_store.similarity_search_with_score_by_vector(motorbike_emb, k=3)

"""
## Search via docstore id
"""
logger.info("## Search via docstore id")

vector_store.index_to_docstore_id

some_docstore_id = 0  # texts[0]

vector_store.docstore._dict[vector_store.index_to_docstore_id[some_docstore_id]]

vector_store.similarity_search_with_score_by_index(some_docstore_id, k=3)

"""
## Save and load
"""
logger.info("## Save and load")

vector_store.save_local("my_annoy_index_and_docstore")

loaded_vector_store = Annoy.load_local(
    "my_annoy_index_and_docstore", embeddings=embeddings_func
)

loaded_vector_store.similarity_search_with_score_by_index(some_docstore_id, k=3)

"""
## Construct from scratch
"""
logger.info("## Construct from scratch")



metadatas = [{"x": "food"}, {"x": "food"}, {"x": "stuff"}, {"x": "animal"}]

embeddings = embeddings_func.embed_documents(texts)

f = len(embeddings[0])

metric = "angular"
index = AnnoyIndex(f, metric=metric)
for i, emb in enumerate(embeddings):
    index.add_item(i, emb)
index.build(10)

documents = []
for i, text in enumerate(texts):
    metadata = metadatas[i] if metadatas else {}
    documents.append(Document(page_content=text, metadata=metadata))
index_to_docstore_id = {i: str(uuid.uuid4()) for i in range(len(documents))}
docstore = InMemoryDocstore(
    {index_to_docstore_id[i]: doc for i, doc in enumerate(documents)}
)

db_manually = Annoy(
    embeddings_func.embed_query, index, metric, docstore, index_to_docstore_id
)

db_manually.similarity_search_with_score("eating!", k=3)

logger.info("\n\n[DONE]", bright=True)