from haystack import Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder, FastembedDocumentEmbedder
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder
from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.components.retrievers.qdrant import QdrantSparseEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from jet.logger import CustomLogger
from transformers import AutoTokenizer
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
# Sparse Embedding Retrieval with Qdrant and FastEmbed


In this notebook, we will see how to use Sparse Embedding Retrieval techniques (such as SPLADE) in Haystack.

We will use the Qdrant Document Store and FastEmbed Sparse Embedders.

## Why SPLADE?

- Sparse Keyword-Based Retrieval (based on BM25 algorithm or similar ones) is simple and fast, requires few resources but relies on lexical matching and struggles to capture semantic meaning.
- Dense Embedding-Based Retrieval takes semantics into account but requires considerable computational resources, usually does not work well on novel domains, and does not consider precise wording.

While good results can be achieved by combining the two approaches ([tutorial](https://haystack.deepset.ai/tutorials/33_hybrid_retrieval)), SPLADE (Sparse Lexical and Expansion Model for Information Retrieval) introduces a new method that encapsulates the positive aspects of both techniques.
In particular, SPLADE uses Language Models like BERT to weigh the relevance of different terms in the query and perform automatic term expansions, reducing the vocabulary mismatch problem (queries and relevant documents often lack term overlap).

Main features:
- Better than dense embedding Retrievers on precise keyword matching
- Better than BM25 on semantic matching
- Slower than BM25
- Still experimental compared to both BM25 and dense embeddings: few models; supported by few Document Stores

**Resources**
- [SPLADE for Sparse Vector Search Explained - great guide by Pinecone](https://www.pinecone.io/learn/splade/)
- [SPLADE GitHub repository, with links to all related papers](https://github.com/naver/splade)

## Install dependencies
"""
logger.info("# Sparse Embedding Retrieval with Qdrant and FastEmbed")

# !pip install -U fastembed-haystack qdrant-haystack wikipedia transformers

"""
## Sparse Embedding Retrieval

### Indexing

#### Create a Qdrant Document Store
"""
logger.info("## Sparse Embedding Retrieval")


document_store = QdrantDocumentStore(
    ":memory:",
    recreate_index=True,
    return_embedding=True,
    use_sparse_embeddings=True  # set this parameter to True, otherwise the collection schema won't allow to store sparse vectors
)

"""
#### Download Wikipedia pages and create raw documents

We download a few Wikipedia pages about animals and create Haystack documents from them.
"""
logger.info("#### Download Wikipedia pages and create raw documents")

nice_animals=["Capybara", "Dolphin"]


raw_docs=[]
for title in nice_animals:
    page = wikipedia.page(title=title, auto_suggest=False)
    doc = Document(content=page.content, meta={"title": page.title, "url":page.url})
    raw_docs.append(doc)

"""
#### Initialize a `FastembedSparseDocumentEmbedder`

The `FastembedSparseDocumentEmbedder` enrichs a list of documents with their sparse embeddings.

We are using `prithvida/Splade_PP_en_v1`, a good sparse embedding model with a permissive license.

We also want to embed the title of the document, because it contains relevant information.

For more customization options, refer to the [docs](https://docs.haystack.deepset.ai/docs/fastembedsparsedocumentembedder).
"""
logger.info("#### Initialize a `FastembedSparseDocumentEmbedder`")


sparse_doc_embedder = FastembedSparseDocumentEmbedder(model="prithvida/Splade_PP_en_v1",
                                                      meta_fields_to_embed=["title"])
sparse_doc_embedder.warm_up()

logger.debug(sparse_doc_embedder.run(documents=[Document(content="An example document")]))

"""
#### Indexing pipeline
"""
logger.info("#### Indexing pipeline")


indexing = Pipeline()
indexing.add_component("cleaner", DocumentCleaner())
indexing.add_component("splitter", DocumentSplitter(split_by='sentence', split_length=4))
indexing.add_component("sparse_doc_embedder", sparse_doc_embedder)
indexing.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE))

indexing.connect("cleaner", "splitter")
indexing.connect("splitter", "sparse_doc_embedder")
indexing.connect("sparse_doc_embedder", "writer")

"""
#### Let's index our documents!
⚠️ If you are running this notebook on Google Colab, please note that Google Colab only provides 2 CPU cores, so the sparse embedding generation could be not as fast as it can be on a standard machine.
"""
logger.info("#### Let's index our documents!")

indexing.run({"documents":raw_docs})

document_store.count_documents()

"""
### Retrieval

#### Retrieval pipeline

Now, we create a simple retrieval Pipeline:
- `FastembedSparseTextEmbedder`: transforms the query into a sparse embedding
- `QdrantSparseEmbeddingRetriever`: looks for relevant documents, based on the similarity of the sparse embeddings
"""
logger.info("### Retrieval")


sparse_text_embedder = FastembedSparseTextEmbedder(model="prithvida/Splade_PP_en_v1")

query_pipeline = Pipeline()
query_pipeline.add_component("sparse_text_embedder", sparse_text_embedder)
query_pipeline.add_component("sparse_retriever", QdrantSparseEmbeddingRetriever(document_store=document_store))

query_pipeline.connect("sparse_text_embedder.sparse_embedding", "sparse_retriever.query_sparse_embedding")

"""
#### Try the retrieval pipeline
"""
logger.info("#### Try the retrieval pipeline")

question = "Where do capybaras live?"

results = query_pipeline.run({"sparse_text_embedder": {"text": question}})


for d in results['sparse_retriever']['documents']:
  rich.logger.debug(f"\nid: {d.id}\n{d.content}\nscore: {d.score}\n---")

"""
## Understanding SPLADE vectors

(Inspiration: [FastEmbed SPLADE notebook](https://qdrant.github.io/fastembed/examples/SPLADE_with_FastEmbed))

We have seen that our model encodes text into a sparse vector (= a vector with many zeros).
An efficient representation of sparse vectors is to save the indices and values of nonzero elements.

Let's try to understand what information resides in these vectors...
"""
logger.info("## Understanding SPLADE vectors")

question = "Where do capybaras live?"
sparse_embedding = sparse_text_embedder.run(text=question)["sparse_embedding"]
rich.logger.debug(sparse_embedding.to_dict())


tokenizer = AutoTokenizer.from_pretrained("Qdrant/Splade_PP_en_v1") # ONNX export of the original model

def get_tokens_and_weights(sparse_embedding, tokenizer):
    token_weight_dict = {}
    for i in range(len(sparse_embedding.indices)):
        token = tokenizer.decode([sparse_embedding.indices[i]])
        weight = sparse_embedding.values[i]
        token_weight_dict[token] = weight

    token_weight_dict = dict(sorted(token_weight_dict.items(), key=lambda item: item[1], reverse=True))
    return token_weight_dict


rich.logger.debug(get_tokens_and_weights(sparse_embedding, tokenizer))

"""
Very nice! 🦫

- tokens are ordered by relevance
- the query is expanded with relevant tokens/terms: "location", "habitat"...

## Hybrid Retrieval

Ideally, techniques like SPLADE are intended to replace other approaches (BM25 and Dense Embedding Retrieval) and their combinations.

However, sometimes it may make sense to combine, for example, Dense Embedding Retrieval and Sparse Embedding Retrieval. You can find some positive examples in the appendix of this paper ([An Analysis of Fusion Functions for Hybrid Retrieval](https://arxiv.org/abs/2210.11934)).
Make sure this works for your use case and conduct an evaluation.

---

Below we show how to create such an application in Haystack.

In the example, we use the Qdrant Hybrid Retriever: it compares dense and sparse query and document embeddings and retrieves the most relevant documents , merging the scores with Reciprocal Rank Fusion.

If you want to customize the behavior more, see Hybrid Retrieval Pipelines ([tutorial](https://haystack.deepset.ai/tutorials/33_hybrid_retrieval)).
"""
logger.info("## Hybrid Retrieval")


document_store = QdrantDocumentStore(
    ":memory:",
    recreate_index=True,
    return_embedding=True,
    use_sparse_embeddings=True,
    embedding_dim = 384
)

hybrid_indexing = Pipeline()
hybrid_indexing.add_component("cleaner", DocumentCleaner())
hybrid_indexing.add_component("splitter", DocumentSplitter(split_by='sentence', split_length=4))
hybrid_indexing.add_component("sparse_doc_embedder", FastembedSparseDocumentEmbedder(model="prithvida/Splade_PP_en_v1", meta_fields_to_embed=["title"]))
hybrid_indexing.add_component("dense_doc_embedder", FastembedDocumentEmbedder(model="BAAI/bge-small-en-v1.5", meta_fields_to_embed=["title"]))
hybrid_indexing.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE))

hybrid_indexing.connect("cleaner", "splitter")
hybrid_indexing.connect("splitter", "sparse_doc_embedder")
hybrid_indexing.connect("sparse_doc_embedder", "dense_doc_embedder")
hybrid_indexing.connect("dense_doc_embedder", "writer")

hybrid_indexing.run({"documents":raw_docs})

document_store.filter_documents()[0]



hybrid_query = Pipeline()
hybrid_query.add_component("sparse_text_embedder", FastembedSparseTextEmbedder(model="prithvida/Splade_PP_en_v1"))
hybrid_query.add_component("dense_text_embedder", FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5", prefix="Represent this sentence for searching relevant passages: "))
hybrid_query.add_component("retriever", QdrantHybridRetriever(document_store=document_store))

hybrid_query.connect("sparse_text_embedder.sparse_embedding", "retriever.query_sparse_embedding")
hybrid_query.connect("dense_text_embedder.embedding", "retriever.query_embedding")

question = "Where do capybaras live?"

results = hybrid_query.run(
    {"dense_text_embedder": {"text": question},
     "sparse_text_embedder": {"text": question}}
)


for d in results['retriever']['documents']:
  rich.logger.debug(f"\nid: {d.id}\n{d.content}\nscore: {d.score}\n---")

"""
## 📚 Docs on Sparse Embedding support in Haystack
- [Retrievers](https://docs.haystack.deepset.ai/docs/retrievers)
- [Qdrant Sparse Embedding Retriever](https://docs.haystack.deepset.ai/docs/qdrantsparseembeddingretriever)
- [Qdrant Hybrid Retriever](https://docs.haystack.deepset.ai/docs/qdranthybridretriever)
- [FastEmbed Sparse Text Embedder](https://docs.haystack.deepset.ai/docs/fastembedsparsetextembedder)
- [Fastembed Sparse Document Embedder](https://docs.haystack.deepset.ai/docs/fastembedsparsedocumentembedder)

(*Notebook by [Stefano Fiorucci](https://github.com/anakin87)*)
"""
logger.info("## 📚 Docs on Sparse Embedding support in Haystack")

logger.info("\n\n[DONE]", bright=True)