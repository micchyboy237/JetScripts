from haystack import Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder, FastembedDocumentEmbedder
from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder, FastembedSparseTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from jet.logger import logger
import os
import rich
import shutil
import wikipedia


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
# Hybrid Retrieval: BM42 + Dense Retrieval

<img src="https://qdrant.tech/articles_data/bm42/preview/title.webp" width="800" style="display:inline;"/>

In this notebook, we will see how to create Hybrid Retrieval pipelines, combining BM42 (a new Sparse embedding Retrieval approach) and Dense embedding Retrieval.

We will use the Qdrant Document Store and Fastembed Embedders.

⚠️ Recent evaluations have raised questions about the validity of BM42. Future developments may address these concerns. Please keep this in mind while reviewing the content.

## Why BM42?

[Qdrant introduced BM42](https://qdrant.tech/articles/bm42/), an algorithm designed to replace BM25 in hybrid RAG pipelines (dense + sparse retrieval).

They found that BM25, while relevant for a long time, has some limitations in common RAG scenarios.

Let's first take a look at BM25 and SPLADE to understand the motivation and the inspiration for BM42.

**BM25**
\begin{equation}
\text{score}(D,Q) = \sum_{i=1}^{N} \text{IDF}(q_i) \times \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}\
\end{equation}


BM25 is an evolution of TF-IDF and has two components:
- Inverse Document Frequency = term importance within a collection
- a component incorporating Term Frequency = term importance within a document

Qdrant folks observed that the TF component relies on document statistics, which only makes sense for longer texts.
This is not the case with common RAG pipelines, where documents are short.

**SPLADE**

Another interesting approach is SPLADE, which uses a BERT-based model to create a bag-of-words representation of the text.
While it generally performs better than BM25, it has some drawbacks:
- tokenization issues with out-of-vocabulary words
- adaptation to new domains requires fine-tuning
- computationally heavy

*For using SPLADE with Haystack, see [this notebook](https://github.com/deepset-ai/haystack-cookbook/blob/main/notebooks/sparse_embedding_retrieval.ipynb).*

**BM42**

\begin{equation}
\text{score}(D,Q) = \sum_{i=1}^{N} \text{IDF}(q_i) \times \text{Attention}(\text{CLS}, q_i)
\end{equation}

Taking inspiration from SPLADE, the Qdrant team developed BM42 to improve BM25.

IDF works well, so they kept it.

But how to quantify term importance within a document?

The attention matrix of Transformer models comes to our aid:
we can the use attention row for the [CLS] token!

To fix tokenization issues, BM42 merges subwords and sums their attention weights.

In their implementation, Qdrant team used all-MiniLM-L6-v2 model, but this technique can work with any Transformer, no fine-tuning needed.


⚠️ Recent evaluations have raised questions about the validity of BM42. Future developments may address these concerns. Please keep this in mind while reviewing the content.

## Install dependencies
"""
logger.info("# Hybrid Retrieval: BM42 + Dense Retrieval")

# !pip install -U fastembed-haystack qdrant-haystack wikipedia transformers

"""
## Hybrid Retrieval

### Indexing

#### Create a Qdrant Document Store
"""
logger.info("## Hybrid Retrieval")


document_store = QdrantDocumentStore(
    ":memory:",
    recreate_index=True,
    embedding_dim=384,
    return_embedding=True,
    use_sparse_embeddings=True,  # set this parameter to True, otherwise the collection schema won't allow to store sparse vectors
    sparse_idf=True  # required for BM42, allows streaming updates of the sparse embeddings while keeping the IDF calculation up-to-date
)

"""
#### Download Wikipedia pages and create raw documents

We download a few Wikipedia pages about animals and create Haystack documents from them.
"""
logger.info("#### Download Wikipedia pages and create raw documents")

nice_animals= ["Capybara", "Dolphin", "Orca", "Walrus"]


raw_docs=[]
for title in nice_animals:
    page = wikipedia.page(title=title, auto_suggest=False)
    doc = Document(content=page.content, meta={"title": page.title, "url":page.url})
    raw_docs.append(doc)

"""
#### Indexing pipeline

Our indexing pipeline includes both a Sparse Document Embedder (based on BM42) and a Dense Document Embedder.
"""
logger.info("#### Indexing pipeline")


hybrid_indexing = Pipeline()
hybrid_indexing.add_component("cleaner", DocumentCleaner())
hybrid_indexing.add_component("splitter", DocumentSplitter(split_by='sentence', split_length=4))
hybrid_indexing.add_component("sparse_doc_embedder", FastembedSparseDocumentEmbedder(model="Qdrant/bm42-all-minilm-l6-v2-attentions", meta_fields_to_embed=["title"]))
hybrid_indexing.add_component("dense_doc_embedder", FastembedDocumentEmbedder(model="BAAI/bge-small-en-v1.5", meta_fields_to_embed=["title"]))
hybrid_indexing.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE))

hybrid_indexing.connect("cleaner", "splitter")
hybrid_indexing.connect("splitter", "sparse_doc_embedder")
hybrid_indexing.connect("sparse_doc_embedder", "dense_doc_embedder")
hybrid_indexing.connect("dense_doc_embedder", "writer")

"""
#### Let's index our documents!
⚠️ If you are running this notebook on Google Colab, please note that Google Colab only provides 2 CPU cores, so the embedding generation with Fastembed could be not as fast as it can be on a standard machine.
"""
logger.info("#### Let's index our documents!")

hybrid_indexing.run({"documents":raw_docs})

document_store.count_documents()

"""
### Retrieval

#### Retrieval pipeline

As already mentioned, BM42 is designed to perform best in Hybrid Retrieval (and Hybrid RAG) pipelines.

- `FastembedSparseTextEmbedder`: transforms the query into a sparse embedding
- `FastembedTextEmbedder`: transforms the query into a dense embedding
- `QdrantHybridRetriever`: looks for relevant documents, based on the similarity of both the embeddings

Qdrant Hybrid Retriever compares dense and sparse query and document embeddings and retrieves the most relevant documents, merging the scores with Reciprocal Rank Fusion.

If you want to customize the fusion behavior more, see Hybrid Retrieval Pipelines ([tutorial](https://haystack.deepset.ai/tutorials/33_hybrid_retrieval)).
"""
logger.info("### Retrieval")



hybrid_query = Pipeline()
hybrid_query.add_component("sparse_text_embedder", FastembedSparseTextEmbedder(model="Qdrant/bm42-all-minilm-l6-v2-attentions"))
hybrid_query.add_component("dense_text_embedder", FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5", prefix="Represent this sentence for searching relevant passages: "))
hybrid_query.add_component("retriever", QdrantHybridRetriever(document_store=document_store, top_k=5))

hybrid_query.connect("sparse_text_embedder.sparse_embedding", "retriever.query_sparse_embedding")
hybrid_query.connect("dense_text_embedder.embedding", "retriever.query_embedding")

"""
#### Try the retrieval pipeline
"""
logger.info("#### Try the retrieval pipeline")

question = "Who eats fish?"

results = hybrid_query.run(
    {"dense_text_embedder": {"text": question},
     "sparse_text_embedder": {"text": question}}
)


for d in results['retriever']['documents']:
  rich.logger.debug(f"\nid: {d.id}\n{d.meta['title']}\n{d.content}\nscore: {d.score}\n---")

question = "capybara social behavior"

results = hybrid_query.run(
    {"dense_text_embedder": {"text": question},
     "sparse_text_embedder": {"text": question}}
)


for d in results['retriever']['documents']:
  rich.logger.debug(f"\nid: {d.id}\n{d.meta['title']}\n{d.content}\nscore: {d.score}\n---")

"""
## 📚 Resources
- [BM42: New Baseline for Hybrid Search - article by Qdrant](https://qdrant.tech/articles/bm42/)
- [Sparse Embedding Retrieval with SPLADE - notebook](https://github.com/deepset-ai/haystack-cookbook/blob/main/notebooks/sparse_embedding_retrieval.ipynb)
- Haystack docs:
  - [Retrievers](https://docs.haystack.deepset.ai/docs/retrievers)
  - [Qdrant Sparse Embedding Retriever](https://docs.haystack.deepset.ai/docs/qdrantsparseembeddingretriever)
  - [Qdrant Hybrid Retriever](https://docs.haystack.deepset.ai/docs/qdranthybridretriever)
  - [FastEmbed Sparse Text Embedder](https://docs.haystack.deepset.ai/docs/fastembedsparsetextembedder)
  - [Fastembed Sparse Document Embedder](https://docs.haystack.deepset.ai/docs/fastembedsparsedocumentembedder)

(*Notebook by [Stefano Fiorucci](https://github.com/anakin87)*)
"""
logger.info("## 📚 Resources")

logger.info("\n\n[DONE]", bright=True)