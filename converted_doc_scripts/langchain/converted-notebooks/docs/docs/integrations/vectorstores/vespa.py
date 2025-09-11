from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import VespaStore
from langchain_text_splitters import CharacterTextSplitter
from vespa.deployment import VespaDocker
from vespa.package import ApplicationPackage, Field, RankProfile
from vespa.package import Component, Parameter
from vespa.package import FieldSet
from vespa.package import HNSW
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
# Vespa

>[Vespa](https://vespa.ai/) is a fully featured search engine and vector database. It supports vector search (ANN), lexical search, and search in structured data, all in the same query.

This notebook shows how to use `Vespa.ai` as a LangChain vector store.

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

In order to create the vector store, we use
[pyvespa](https://pyvespa.readthedocs.io/en/latest/index.html) to create a
connection a `Vespa` service.
"""
logger.info("# Vespa")

# %pip install --upgrade --quiet  pyvespa

"""
Using the `pyvespa` package, you can either connect to a
[Vespa Cloud instance](https://pyvespa.readthedocs.io/en/latest/deploy-vespa-cloud.html)
or a local
[Docker instance](https://pyvespa.readthedocs.io/en/latest/deploy-docker.html).
Here, we will create a new Vespa application and deploy that using Docker.

#### Creating a Vespa application

First, we need to create an application package:
"""
logger.info("#### Creating a Vespa application")


app_package = ApplicationPackage(name="testapp")
app_package.schema.add_fields(
    Field(
        name="text", type="string", indexing=["index", "summary"], index="enable-bm25"
    ),
    Field(
        name="embedding",
        type="tensor<float>(x[384])",
        indexing=["attribute", "summary"],
        attribute=["distance-metric: angular"],
    ),
)
app_package.schema.add_rank_profile(
    RankProfile(
        name="default",
        first_phase="closeness(field, embedding)",
        inputs=[("query(query_embedding)", "tensor<float>(x[384])")],
    )
)

"""
This sets up a Vespa application with a schema for each document that contains
two fields: `text` for holding the document text and `embedding` for holding
the embedding vector. The `text` field is set up to use a BM25 index for
efficient text retrieval, and we'll see how to use this and hybrid search a
bit later.

The `embedding` field is set up with a vector of length 384 to hold the
embedding representation of the text. See
[Vespa's Tensor Guide](https://docs.vespa.ai/en/tensor-user-guide.html)
for more on tensors in Vespa.

Lastly, we add a [rank profile](https://docs.vespa.ai/en/ranking.html) to
instruct Vespa how to order documents. Here we set this up with a
[nearest neighbor search](https://docs.vespa.ai/en/nearest-neighbor-search.html).

Now we can deploy this application locally:
"""
logger.info("This sets up a Vespa application with a schema for each document that contains")


vespa_docker = VespaDocker()
vespa_app = vespa_docker.deploy(application_package=app_package)

"""
This deploys and creates a connection to a `Vespa` service. In case you
already have a Vespa application running, for instance in the cloud,
please refer to the PyVespa application for how to connect.

#### Creating a Vespa vector store

Now, let's load some documents:
"""
logger.info("#### Creating a Vespa vector store")


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

"""
Here, we also set up local sentence embedder to transform the text to embedding
vectors. One could also use Ollama embeddings, but the vector length needs to
be updated to `1536` to reflect the larger size of that embedding.

To feed these to Vespa, we need to configure how the vector store should map to
fields in the Vespa application. Then we create the vector store directly from
this set of documents:
"""
logger.info("Here, we also set up local sentence embedder to transform the text to embedding")

vespa_config = dict(
    page_content_field="text",
    embedding_field="embedding",
    input_field="query_embedding",
)


db = VespaStore.from_documents(docs, embedding_function, app=vespa_app, **vespa_config)

"""
This creates a Vespa vector store and feeds that set of documents to Vespa.
The vector store takes care of calling the embedding function for each document
and inserts them into the database.

We can now query the vector store:
"""
logger.info("This creates a Vespa vector store and feeds that set of documents to Vespa.")

query = "What did the president say about Ketanji Brown Jackson"
results = db.similarity_search(query)

logger.debug(results[0].page_content)

"""
This will use the embedding function given above to create a representation
for the query and use that to search Vespa. Note that this will use the
`default` ranking function, which we set up in the application package
above. You can use the `ranking` argument to `similarity_search` to
specify which ranking function to use.

Please refer to the [pyvespa documentation](https://pyvespa.readthedocs.io/en/latest/getting-started-pyvespa.html#Query)
for more information.

This covers the basic usage of the Vespa store in LangChain.
Now you can return the results and continue using these in LangChain.

#### Updating documents

An alternative to calling `from_documents`, you can create the vector
store directly and call `add_texts` from that. This can also be used to update
documents:
"""
logger.info("#### Updating documents")

query = "What did the president say about Ketanji Brown Jackson"
results = db.similarity_search(query)
result = results[0]

result.page_content = "UPDATED: " + result.page_content
db.add_texts([result.page_content], [result.metadata], result.metadata["id"])

results = db.similarity_search(query)
logger.debug(results[0].page_content)

"""
However, the `pyvespa` library contains methods to manipulate
content on Vespa which you can use directly.

#### Deleting documents

You can delete documents using the `delete` function:
"""
logger.info("#### Deleting documents")

result = db.similarity_search(query)

db.delete(["32"])
result = db.similarity_search(query)

"""
Again, the `pyvespa` connection contains methods to delete documents as well.

### Returning with scores

The `similarity_search` method only returns the documents in order of
relevancy. To retrieve the actual scores:
"""
logger.info("### Returning with scores")

results = db.similarity_search_with_score(query)
result = results[0]

"""
This is a result of using the `"all-MiniLM-L6-v2"` embedding model using the
cosine distance function (as given by the argument `angular` in the
application function).

Different embedding functions need different distance functions, and Vespa
needs to know which distance function to use when orderings documents.
Please refer to the
[documentation on distance functions](https://docs.vespa.ai/en/reference/schema-reference.html#distance-metric)
for more information.

### As retriever

To use this vector store as a
[LangChain retriever](/docs/how_to#retrievers)
simply call the `as_retriever` function, which is a standard vector store
method:
"""
logger.info("### As retriever")

db = VespaStore.from_documents(docs, embedding_function, app=vespa_app, **vespa_config)
retriever = db.as_retriever()
query = "What did the president say about Ketanji Brown Jackson"
results = retriever.invoke(query)

"""
This allows for more general, unstructured, retrieval from the vector store.

### Metadata

In the example so far, we've only used the text and the embedding for that
text. Documents usually contain additional information, which in LangChain
is referred to as metadata.

Vespa can contain many fields with different types by adding them to the application
package:
"""
logger.info("### Metadata")

app_package.schema.add_fields(
    Field(name="date", type="string", indexing=["attribute", "summary"]),
    Field(name="rating", type="int", indexing=["attribute", "summary"]),
    Field(name="author", type="string", indexing=["attribute", "summary"]),
)
vespa_app = vespa_docker.deploy(application_package=app_package)

"""
We can add some metadata fields in the documents:
"""
logger.info("We can add some metadata fields in the documents:")

for i, doc in enumerate(docs):
    doc.metadata["date"] = f"2023-{(i % 12) + 1}-{(i % 28) + 1}"
    doc.metadata["rating"] = range(1, 6)[i % 5]
    doc.metadata["author"] = ["Joe Biden", "Unknown"][min(i, 1)]

"""
And let the Vespa vector store know about these fields:
"""
logger.info("And let the Vespa vector store know about these fields:")

vespa_config.update(dict(metadata_fields=["date", "rating", "author"]))

"""
Now, when searching for these documents, these fields will be returned.
Also, these fields can be filtered on:
"""
logger.info("Now, when searching for these documents, these fields will be returned.")

db = VespaStore.from_documents(docs, embedding_function, app=vespa_app, **vespa_config)
query = "What did the president say about Ketanji Brown Jackson"
results = db.similarity_search(query, filter="rating > 3")

"""
### Custom query

If the default behavior of the similarity search does not fit your
requirements, you can always provide your own query. Thus, you don't
need to provide all of the configuration to the vector store, but
rather just write this yourself.

First, let's add a BM25 ranking function to our application:
"""
logger.info("### Custom query")


app_package.schema.add_field_set(FieldSet(name="default", fields=["text"]))
app_package.schema.add_rank_profile(RankProfile(name="bm25", first_phase="bm25(text)"))
vespa_app = vespa_docker.deploy(application_package=app_package)
db = VespaStore.from_documents(docs, embedding_function, app=vespa_app, **vespa_config)

"""
Then, to perform a regular text search based on BM25:
"""
logger.info("Then, to perform a regular text search based on BM25:")

query = "What did the president say about Ketanji Brown Jackson"
custom_query = {
    "yql": "select * from sources * where userQuery()",
    "query": query,
    "type": "weakAnd",
    "ranking": "bm25",
    "hits": 4,
}
results = db.similarity_search_with_score(query, custom_query=custom_query)

"""
All of the powerful search and query capabilities of Vespa can be used
by using a custom query. Please refer to the Vespa documentation on it's
[Query API](https://docs.vespa.ai/en/query-api.html) for more details.

### Hybrid search

Hybrid search means using both a classic term-based search such as
BM25 and a vector search and combining the results. We need to create
a new rank profile for hybrid search on Vespa:
"""
logger.info("### Hybrid search")

app_package.schema.add_rank_profile(
    RankProfile(
        name="hybrid",
        first_phase="log(bm25(text)) + 0.5 * closeness(field, embedding)",
        inputs=[("query(query_embedding)", "tensor<float>(x[384])")],
    )
)
vespa_app = vespa_docker.deploy(application_package=app_package)
db = VespaStore.from_documents(docs, embedding_function, app=vespa_app, **vespa_config)

"""
Here, we score each document as a combination of it's BM25 score and its
distance score. We can query using a custom query:
"""
logger.info("Here, we score each document as a combination of it's BM25 score and its")

query = "What did the president say about Ketanji Brown Jackson"
query_embedding = embedding_function.embed_query(query)
nearest_neighbor_expression = (
    "{targetHits: 4}nearestNeighbor(embedding, query_embedding)"
)
custom_query = {
    "yql": f"select * from sources * where {nearest_neighbor_expression} and userQuery()",
    "query": query,
    "type": "weakAnd",
    "input.query(query_embedding)": query_embedding,
    "ranking": "hybrid",
    "hits": 4,
}
results = db.similarity_search_with_score(query, custom_query=custom_query)

"""
### Native embedders in Vespa

Up until this point we've used an embedding function in Python to provide
embeddings for the texts. Vespa supports embedding function natively, so
you can defer this calculation in to Vespa. One benefit is the ability to use
GPUs when embedding documents if you have a large collections.

Please refer to [Vespa embeddings](https://docs.vespa.ai/en/embedding.html)
for more information.

First, we need to modify our application package:
"""
logger.info("### Native embedders in Vespa")


app_package.components = [
    Component(
        id="hf-embedder",
        type="hugging-face-embedder",
        parameters=[
            Parameter("transformer-model", {"path": "..."}),
            Parameter("tokenizer-model", {"url": "..."}),
        ],
    )
]
Field(
    name="hfembedding",
    type="tensor<float>(x[384])",
    is_document_field=False,
    indexing=["input text", "embed hf-embedder", "attribute", "summary"],
    attribute=["distance-metric: angular"],
)
app_package.schema.add_rank_profile(
    RankProfile(
        name="hf_similarity",
        first_phase="closeness(field, hfembedding)",
        inputs=[("query(query_embedding)", "tensor<float>(x[384])")],
    )
)

"""
Please refer to the embeddings documentation on adding embedder models
and tokenizers to the application. Note that the `hfembedding` field
includes instructions for embedding using the `hf-embedder`.

Now we can query with a custom query:
"""
logger.info("Please refer to the embeddings documentation on adding embedder models")

query = "What did the president say about Ketanji Brown Jackson"
nearest_neighbor_expression = (
    "{targetHits: 4}nearestNeighbor(internalembedding, query_embedding)"
)
custom_query = {
    "yql": f"select * from sources * where {nearest_neighbor_expression}",
    "input.query(query_embedding)": f'embed(hf-embedder, "{query}")',
    "ranking": "internal_similarity",
    "hits": 4,
}
results = db.similarity_search_with_score(query, custom_query=custom_query)

"""
Note that the query here includes an `embed` instruction to embed the query
using the same model as for the documents.

### Approximate nearest neighbor

In all of the above examples, we've used exact nearest neighbor to
find results. However, for large collections of documents this is
not feasible as one has to scan through all documents to find the
best matches. To avoid this, we can use
[approximate nearest neighbors](https://docs.vespa.ai/en/approximate-nn-hnsw.html).

First, we can change the embedding field to create a HNSW index:
"""
logger.info("### Approximate nearest neighbor")


app_package.schema.add_fields(
    Field(
        name="embedding",
        type="tensor<float>(x[384])",
        indexing=["attribute", "summary", "index"],
        ann=HNSW(
            distance_metric="angular",
            max_links_per_node=16,
            neighbors_to_explore_at_insert=200,
        ),
    )
)

"""
This creates a HNSW index on the embedding data which allows for efficient
searching. With this set, we can easily search using ANN by setting
the `approximate` argument to `True`:
"""
logger.info("This creates a HNSW index on the embedding data which allows for efficient")

query = "What did the president say about Ketanji Brown Jackson"
results = db.similarity_search(query, approximate=True)

"""
This covers most of the functionality in the Vespa vector store in LangChain.
"""
logger.info("This covers most of the functionality in the Vespa vector store in LangChain.")

logger.info("\n\n[DONE]", bright=True)