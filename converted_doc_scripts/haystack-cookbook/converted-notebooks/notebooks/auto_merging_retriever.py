from haystack import Document
from haystack import Pipeline
from haystack.components.preprocessors import HierarchicalDocumentSplitter
from haystack.components.retrievers import AutoMergingRetriever
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from jet.logger import logger
from typing import List
from typing import Tuple
import csv
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
# Improving Retrieval with Auto-Merging and Hierarchical Document Retrieval

This notebook shows how to use Haystack components: `AutoMergingRetriever` and `HierarchicalDocumentSplitter`.

- 📚[Read the full article here](https://haystack.deepset.ai/blog/improve-retrieval-with-auto-merging)

## Setting up
"""
logger.info("# Improving Retrieval with Auto-Merging and Hierarchical Document Retrieval")

# !pip install haystack-ai

"""
## Let's get a dataset to index and explore

- We will use a dataset containing 2225 new articles part of the paper by "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006. by D. Greene and P. Cunningham.

- The original dataset is available at http://mlg.ucd.ie/datasets/bbc.html, but we will instead use a CSV processed version available here: https://raw.githubusercontent.com/amankharwal/Website-data/master/bbc-news-data.csv
"""
logger.info("## Let's get a dataset to index and explore")

# !wget https://raw.githubusercontent.com/amankharwal/Website-data/master/bbc-news-data.csv

"""
## Let's convert the raw data into Haystack Documents
"""
logger.info("## Let's convert the raw data into Haystack Documents")


def read_documents() -> List[Document]:
    with open("bbc-news-data.csv", "r") as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader, None)  # skip the headers
        documents = []
        for row in reader:
            category = row[0].strip()
            title = row[2].strip()
            text = row[3].strip()
            documents.append(Document(content=text, meta={"category": category, "title": title}))

    return documents

docs = read_documents()

docs[0:5]

"""
We can see that we have successfully created Documents.

## Document Splitting and Indexing

Now we split each document into smaller ones creating an hierarchical document structure connecting each smaller child documents with the corresponding parent document.

We also create two document stores, one for the leaf documents and the other for the parent documents.
"""
logger.info("## Document Splitting and Indexing")




def indexing(documents: List[Document]) -> Tuple[InMemoryDocumentStore, InMemoryDocumentStore]:
    splitter = HierarchicalDocumentSplitter(block_sizes={10, 3}, split_overlap=0, split_by="word")
    docs = splitter.run(documents)

    leaf_documents = [doc for doc in docs["documents"] if doc.meta["__level"] == 1]
    leaf_doc_store = InMemoryDocumentStore()
    leaf_doc_store.write_documents(leaf_documents, policy=DuplicatePolicy.OVERWRITE)

    parent_documents = [doc for doc in docs["documents"] if doc.meta["__level"] == 0]
    parent_doc_store = InMemoryDocumentStore()
    parent_doc_store.write_documents(parent_documents, policy=DuplicatePolicy.OVERWRITE)

    return leaf_doc_store, parent_doc_store

leaf_doc_store, parent_doc_store = indexing(docs)

"""
## Retrieving Documents with Auto-Merging

We are now ready to query the document store using the `AutoMergingRetriever`. Let's build a pipeline that uses the `BM25Retriever` to handle the user queries, and we connect it to the `AutoMergingRetriever`, which, based on the documents retrieved and the hierarchical structure, decides whether the leaf documents or the parent document is returned.
"""
logger.info("## Retrieving Documents with Auto-Merging")


def querying_pipeline(leaf_doc_store: InMemoryDocumentStore, parent_doc_store: InMemoryDocumentStore, threshold: float = 0.6):
    pipeline = Pipeline()
    bm25_retriever = InMemoryBM25Retriever(document_store=leaf_doc_store)
    auto_merge_retriever = AutoMergingRetriever(parent_doc_store, threshold=threshold)
    pipeline.add_component(instance=bm25_retriever, name="BM25Retriever")
    pipeline.add_component(instance=auto_merge_retriever, name="AutoMergingRetriever")
    pipeline.connect("BM25Retriever.documents", "AutoMergingRetriever.documents")
    return pipeline

"""
Let's create this pipeline by setting the threshold for the `AutoMergingRetriever` at 0.6
"""
logger.info("Let's create this pipeline by setting the threshold for the `AutoMergingRetriever` at 0.6")

pipeline = querying_pipeline(leaf_doc_store, parent_doc_store, threshold=0.6)

"""
Let's now query the pipeline for document store for articles related to cybersecurity. Let's also make use of the pipeline parameter `include_outputs_from` to also get the outputs from the `BM25Retriever` component.
"""
logger.info("Let's now query the pipeline for document store for articles related to cybersecurity. Let's also make use of the pipeline parameter `include_outputs_from` to also get the outputs from the `BM25Retriever` component.")

result = pipeline.run(data={'query': 'phishing attacks spoof websites spam e-mails spyware'},  include_outputs_from={'BM25Retriever'})

len(result['AutoMergingRetriever']['documents'])

len(result['BM25Retriever']['documents'])

retrieved_doc_titles_bm25 = sorted([d.meta['title'] for d in result['BM25Retriever']['documents']])

retrieved_doc_titles_bm25

retrieved_doc_titles_automerging = sorted([d.meta['title'] for d in result['AutoMergingRetriever']['documents']])

retrieved_doc_titles_automerging

logger.info("\n\n[DONE]", bright=True)