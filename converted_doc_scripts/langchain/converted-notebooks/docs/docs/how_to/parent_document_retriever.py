from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
# How to use the Parent Document Retriever

When splitting documents for [retrieval](/docs/concepts/retrieval/), there are often conflicting desires:

1. You may want to have small documents, so that their embeddings can most
    accurately reflect their meaning. If too long, then the embeddings can
    lose meaning.
2. You want to have long enough documents that the context of each chunk is
    retained.

The `ParentDocumentRetriever` strikes that balance by splitting and storing
small chunks of data. During retrieval, it first fetches the small chunks
but then looks up the parent ids for those chunks and returns those larger
documents.

Note that "parent document" refers to the document that a small chunk
originated from. This can either be the whole raw document OR a larger
chunk.
"""
logger.info("# How to use the Parent Document Retriever")


loaders = [
    TextLoader("paul_graham_essay.txt"),
    TextLoader("state_of_the_union.txt"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

"""
## Retrieving full documents

In this mode, we want to retrieve the full documents. Therefore, we only specify a child [splitter](/docs/concepts/text_splitters/).
"""
logger.info("## Retrieving full documents")

child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=OllamaEmbeddings(model="mxbai-embed-large")
)
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

retriever.add_documents(docs, ids=None)

"""
This should yield two keys, because we added two documents.
"""
logger.info("This should yield two keys, because we added two documents.")

list(store.yield_keys())

"""
Let's now call the vector store search functionality - we should see that it returns small chunks (since we're storing the small chunks).
"""
logger.info("Let's now call the vector store search functionality - we should see that it returns small chunks (since we're storing the small chunks).")

sub_docs = vectorstore.similarity_search("justice breyer")

logger.debug(sub_docs[0].page_content)

"""
Let's now retrieve from the overall retriever. This should return large documents - since it returns the documents where the smaller chunks are located.
"""
logger.info("Let's now retrieve from the overall retriever. This should return large documents - since it returns the documents where the smaller chunks are located.")

retrieved_docs = retriever.invoke("justice breyer")

len(retrieved_docs[0].page_content)

"""
## Retrieving larger chunks

Sometimes, the full documents can be too big to want to retrieve them as is. In that case, what we really want to do is to first split the raw documents into larger chunks, and then split it into smaller chunks. We then index the smaller chunks, but on retrieval we retrieve the larger chunks (but still not the full documents).
"""
logger.info("## Retrieving larger chunks")

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
vectorstore = Chroma(
    collection_name="split_parents", embedding_function=OllamaEmbeddings(model="mxbai-embed-large")
)
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(docs)

"""
We can see that there are much more than two documents now - these are the larger chunks.
"""
logger.info(
    "We can see that there are much more than two documents now - these are the larger chunks.")

len(list(store.yield_keys()))

"""
Let's make sure the underlying vector store still retrieves the small chunks.
"""
logger.info(
    "Let's make sure the underlying vector store still retrieves the small chunks.")

sub_docs = vectorstore.similarity_search("justice breyer")

logger.debug(sub_docs[0].page_content)

retrieved_docs = retriever.invoke("justice breyer")

len(retrieved_docs[0].page_content)

logger.debug(retrieved_docs[0].page_content)

logger.info("\n\n[DONE]", bright=True)
