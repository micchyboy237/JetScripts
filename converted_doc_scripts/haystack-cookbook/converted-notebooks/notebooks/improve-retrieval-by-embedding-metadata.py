from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import ComponentDevice
from jet.logger import CustomLogger
import os
import rich
import shutil
import wikipedia


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
#  🔍 Improve retrieval by embedding meaningful metadata 🏷️


*Notebook by [Stefano Fiorucci](https://github.com/anakin87)*

In this notebook, I do some experiments on embedding meaningful metadata to improve Document retrieval.
"""
logger.info("#  🔍 Improve retrieval by embedding meaningful metadata 🏷️")

# %%capture
# ! pip install wikipedia haystack-ai sentence_transformers rich


"""
## Load data from Wikipedia

We are going to download the Wikipedia pages related to some bands, using the python library `wikipedia`.

These pages are converted into Haystack Documents.
"""
logger.info("## Load data from Wikipedia")

some_bands="""The Beatles
Rolling stones
Dire Straits
The Cure
The Smiths""".split("\n")


raw_docs=[]

for title in some_bands:
    page = wikipedia.page(title=title, auto_suggest=False)
    doc = Document(content=page.content, meta={"title": page.title, "url":page.url})
    raw_docs.append(doc)

"""
## 🔧 Setup the experiment

### Utility functions to create Pipelines

The **indexing Pipeline** transforms the Documents and stores them (with vectors) in a Document Store. The **retrieval Pipeline** takes a query as input and perform the vector search.


I build some utility functions to create different indexing and retrieval Pipelines.

In fact, I am interested in comparing the standard approach (where we only embed text) with the embedding metadata strategy (we embed text + meaningful metadata).
"""
logger.info("## 🔧 Setup the experiment")


def create_indexing_pipeline(document_store, metadata_fields_to_embed):

  indexing = Pipeline()
  indexing.add_component("cleaner", DocumentCleaner())
  indexing.add_component("splitter", DocumentSplitter(split_by='sentence', split_length=2))

  indexing.add_component("doc_embedder", SentenceTransformersDocumentEmbedder(model="thenlper/gte-large",
                                                                              device=ComponentDevice.from_str("cuda:0"),
                                                                              meta_fields_to_embed=metadata_fields_to_embed)
  )
  indexing.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE))

  indexing.connect("cleaner", "splitter")
  indexing.connect("splitter", "doc_embedder")
  indexing.connect("doc_embedder", "writer")

  return indexing

def create_retrieval_pipeline(document_store):

  retrieval = Pipeline()
  retrieval.add_component("text_embedder", SentenceTransformersTextEmbedder(model="thenlper/gte-large",
                                                                            device=ComponentDevice.from_str("cuda:0")))
  retrieval.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, scale_score=False, top_k=3))

  retrieval.connect("text_embedder", "retriever")

  return retrieval

"""
###  Create the Pipelines

Let's define 2 Document Stores, to compare the different approaches.
"""
logger.info("###  Create the Pipelines")

document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
document_store_w_embedded_metadata = InMemoryDocumentStore(embedding_similarity_function="cosine")

"""
Now, I create the 2 indexing pipelines and run them.
"""
logger.info("Now, I create the 2 indexing pipelines and run them.")

indexing_pipe_std = create_indexing_pipeline(document_store=document_store, metadata_fields_to_embed=[])

indexing_pipe_w_embedded_metadata = create_indexing_pipeline(document_store=document_store_w_embedded_metadata, metadata_fields_to_embed=["title"])

indexing_pipe_std.run({"cleaner":{"documents":raw_docs}})
indexing_pipe_w_embedded_metadata.run({"cleaner":{"documents":raw_docs}})

logger.debug(len(document_store.filter_documents()))
logger.debug(len(document_store_w_embedded_metadata.filter_documents()))

"""
Create the 2 retrieval pipelines.
"""
logger.info("Create the 2 retrieval pipelines.")

retrieval_pipe_std = create_retrieval_pipeline(document_store=document_store)

retrieval_pipe_w_embedded_metadata = create_retrieval_pipeline(document_store=document_store_w_embedded_metadata)

"""
## 🧪 Run the experiment!
"""
logger.info("## 🧪 Run the experiment!")

res=retrieval_pipe_std.run({"text_embedder":{"text":"have the beatles ever been to bangor?"}})
for doc in res['retriever']['documents']:
  rich.logger.debug(doc)
  rich.logger.debug(doc.content+"\n")

"""
❌ the retrieved Documents seem irrelevant
"""

res=retrieval_pipe_w_embedded_metadata.run({"text_embedder":{"text":"have the beatles ever been to bangor?"}})
for doc in res['retriever']['documents']:
  rich.logger.debug(doc)
  rich.logger.debug(doc.content+"\n")

"""
✅ the first Document is relevant
"""

res=retrieval_pipe_std.run({"text_embedder":{"text":"What announcements did the band The Cure make in 2022?"}})
for doc in res['retriever']['documents']:
  rich.logger.debug(doc)
  rich.logger.debug(doc.content)

"""
❌ the retrieved Documents seem irrelevant
"""

res=retrieval_pipe_w_embedded_metadata.run({"text_embedder":{"text":"What announcements did the band The Cure make in 2022?"}})
for doc in res['retriever']['documents']:
  rich.logger.debug(doc)
  rich.logger.debug(doc.content)

"""
✅ some Documents are relevant

## ⚠️ Notes of caution

- This technique is not a silver bullet
- It works well when the embedded metadata are meaningful and distinctive
- I would say that the embedded metadata should be meaningful from the perspective of the embedding model. For example, I don't expect embedding numbers to work well.
"""
logger.info("## ⚠️ Notes of caution")

logger.info("\n\n[DONE]", bright=True)