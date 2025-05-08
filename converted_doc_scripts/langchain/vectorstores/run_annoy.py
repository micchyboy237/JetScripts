from typing import Callable, Union
from shared.data_types.job import JobData
from jet.file import load_file
from jet.transformers.formatters import format_json
from jet.logger import logger
from jet.utils.object import extract_values_by_paths
from langchain_community.vectorstores import Annoy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import uuid
from annoy import AnnoyIndex
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document


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


def log_response(results: Union[list[Document], list[tuple[Document, float]]]):
    """Logs the response results in a readable format."""
    if isinstance(results, list) and results:
        # Case for functions returning List[Tuple[Document, float]]
        if isinstance(results[0], tuple):
            # logged_response = [(doc.page_content, score) for doc, score in results]
            logged_response = [
                {"score": score, "text": doc.page_content}
                for doc, score in results
            ]
        else:  # Case for functions returning List[Document]
            # logged_response = [doc.page_content for doc in results]
            logged_response = [
                {"score": None, "text": doc.page_content}
                for doc in results
            ]
    else:
        # logged_response = results
        logged_response = [
            {"score": None, "text": doc.page_content}
            for doc in results
        ]

    logger.info(f"Logged response: {type(logged_response)}")
    for result in logged_response:
        if result.get('score'):
            logger.log(" -", f"{result['text'][:25]}:",
                       f"{result['score']:.4f}", colors=["GRAY", "WHITE", "SUCCESS"])
        else:
            logger.log(" -", f"{result['text'][:25]}:",
                       colors=["GRAY", "WHITE"])


# %pip install --upgrade --quiet  annoy

data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
output_dir = "generated/my_annoy_index_and_docstore"
data: list[JobData] = load_file(data_file)
# items = [
#     {"id": item['id'], "text": f"Job Title: {item['title']}\n\n{item['details']}"}
#     if item['title'] not in item['details'].split("\n")[0]
#     else {"id": item['id'], "text": item['details']}
#     for item in data
#     # if not item.get('entities')
# ]
# docs = [Document(page_content=item['text']) for item in items]
# texts = [item['text'] for item in items]

text_attributes = [
    "title",
    "details",
]
metadata_attributes = [
    "id",
    "link",
    "company",
    "posted_date",
    "salary",
    "job_type",
    "hours_per_week",
    "domain",
    "tags",
    "keywords",
    "entities.role",
    "entities.application",
    "entities.technology_stack",
    "entities.qualifications",
]


docs: list[Document] = []
for item in data:
    title = item['title']
    details = item['details']

    textdata_str = f"Details:\n{details}"
    if title not in details:
        textdata_str = f"Job Title: {title}\n{textdata_str}"

    metadata = extract_values_by_paths(
        item, metadata_attributes, is_flattened=True)

    docs.append(Document(
        page_content=textdata_str,
        metadata=metadata,
    ))
texts = [doc.page_content for doc in docs]
metadatas = [doc.metadata for doc in docs]

query = "Which uses React.js for Web development?"


"""
## Create VectorStore from texts
"""

logger.newline()
logger.orange("## Create VectorStore from texts")

model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings_func = HuggingFaceEmbeddings(model_name=model_name)

# texts = ["pizza is great", "I love salad", "my car", "a dog"]

vector_store = Annoy.from_texts(texts, embeddings_func)

vector_store_v2 = Annoy.from_texts(
    texts, embeddings_func, metric="dot", n_trees=100, n_jobs=1
)

# query = "food"

results = vector_store.similarity_search(query, k=3)
logger.info("Result 1 (similarity_search):")
logger.log("Query:", query, colors=["GRAY", "DEBUG"])
log_response(results)

results = vector_store.similarity_search_with_score(query, k=3)
logger.info("Result 2 (similarity_search_with_score):")
logger.log("Query:", query, colors=["GRAY", "DEBUG"])
log_response(results)

"""
## Create VectorStore from docs
"""

logger.newline()
logger.orange("## Create VectorStore from docs")

# loader = TextLoader("../../how_to/state_of_the_union.txtn.txtn.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# docs[:5]

vector_store_from_docs = Annoy.from_documents(docs, embeddings_func)

# query = "Which uses React.js for web development?"
results = vector_store_from_docs.similarity_search(query)
logger.info("Result 1 (similarity_search):")
logger.log("Query:", query, colors=["GRAY", "DEBUG"])
log_response(results)


"""
## Create VectorStore via existing embeddings
"""

logger.newline()
logger.orange("## Create VectorStore via existing embeddings")

embs = embeddings_func.embed_documents(texts)

data = list(zip(texts, embs))

vector_store_from_embeddings = Annoy.from_embeddings(data, embeddings_func)

# query = "food"
results = vector_store_from_embeddings.similarity_search_with_score(query, k=3)
logger.info("Result 1 (similarity_search_with_score):")
logger.log("Query:", query, colors=["GRAY", "DEBUG"])
log_response(results)


"""
## Search via embeddings
"""

logger.newline()
logger.orange("## Search via embeddings")

# query = "motorbike"
motorbike_emb = embeddings_func.embed_query(query)

results = vector_store.similarity_search_by_vector(motorbike_emb, k=3)
logger.info("Result 1 (similarity_search_with_score):")
logger.log("Query:", query, colors=["GRAY", "DEBUG"])
log_response(results)

results = vector_store.similarity_search_with_score_by_vector(
    motorbike_emb, k=3)
logger.info("Result 2 (similarity_search_with_score_by_vector):")
logger.log("Query:", query, colors=["GRAY", "DEBUG"])
log_response(results)

"""
## Search via docstore id
"""

logger.newline()
logger.orange("## Search via docstore id")

logger.log("vector_store.index_to_docstore_id:",
           vector_store.index_to_docstore_id, colors=["GRAY", "DEBUG"])

some_docstore_id = 0  # texts[0]

vector_store.docstore._dict[vector_store.index_to_docstore_id[some_docstore_id]]

results = vector_store.similarity_search_with_score_by_index(
    some_docstore_id, k=3)
logger.info("Result 1 (similarity_search_with_score_by_index):")
logger.log("Query:", some_docstore_id, colors=["GRAY", "DEBUG"])
log_response(results)

"""
## Save and load
"""

logger.newline()
logger.orange("## Save and load")

vector_store.save_local(output_dir)

loaded_vector_store = Annoy.load_local(
    output_dir, embeddings=embeddings_func, allow_dangerous_deserialization=True
)

results = loaded_vector_store.similarity_search_with_score_by_index(
    some_docstore_id, k=3)

logger.info("Result 1 (similarity_search_with_score_by_index):")
logger.log("Query:", some_docstore_id, colors=["GRAY", "DEBUG"])
log_response(results)

"""
## Construct from scratch
"""

logger.newline()
logger.orange("## Construct from scratch")

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

# query = "eating!"
results = db_manually.similarity_search_with_score(query, k=3)
logger.info("Result 1 (similarity_search_with_score):")
logger.log("Query:", query, colors=["GRAY", "DEBUG"])
log_response(results)

logger.info("\n\n[DONE]", bright=True)
