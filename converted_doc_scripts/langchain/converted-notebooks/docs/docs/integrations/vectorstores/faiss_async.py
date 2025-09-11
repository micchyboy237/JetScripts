from jet.models.config import MODELS_CACHE_DIR
from jet.transformers.formatters import format_json
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
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
# Faiss (Async)

>[Facebook AI Similarity Search (Faiss)](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also includes supporting code for evaluation and parameter tuning.
>
>See [The FAISS Library](https://arxiv.org/pdf/2401.08281) paper.

[Faiss documentation](https://faiss.ai/).

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

This notebook shows how to use functionality related to the `FAISS` vector database using `asyncio`.
LangChain implemented the synchronous and asynchronous vector store functions.

See `synchronous` version [here](/docs/integrations/vectorstores/faiss).
"""
logger.info("# Faiss (Async)")

# %pip install --upgrade --quiet  faiss-gpu # For CUDA 7.5+ Supported GPU's.
# %pip install --upgrade --quiet  faiss-cpu # For CPU Installation

"""
We want to use OllamaEmbeddings so we have to get the Ollama API Key.
"""
logger.info("We want to use OllamaEmbeddings so we have to get the Ollama API Key.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")



loader = TextLoader("../../../extras/modules/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db = await FAISS.afrom_documents(docs, embeddings)
logger.success(format_json(db))

query = "What did the president say about Ketanji Brown Jackson"
docs = await db.asimilarity_search(query)
logger.success(format_json(docs))

logger.debug(docs[0].page_content)

"""
## Similarity Search with score
There are some FAISS specific methods. One of them is `similarity_search_with_score`, which allows you to return not only the documents but also the distance score of the query to them. The returned distance score is L2 distance. Therefore, a lower score is better.
"""
logger.info("## Similarity Search with score")

docs_and_scores = await db.asimilarity_search_with_score(query)
logger.success(format_json(docs_and_scores))

docs_and_scores[0]

"""
It is also possible to do a search for documents similar to a given embedding vector using `similarity_search_by_vector` which accepts an embedding vector as a parameter instead of a string.
"""
logger.info("It is also possible to do a search for documents similar to a given embedding vector using `similarity_search_by_vector` which accepts an embedding vector as a parameter instead of a string.")

embedding_vector = await embeddings.aembed_query(query)
logger.success(format_json(embedding_vector))
docs_and_scores = await db.asimilarity_search_by_vector(embedding_vector)
logger.success(format_json(docs_and_scores))

"""
## Saving and loading
You can also save and load a FAISS index. This is useful so you don't have to recreate it everytime you use it.
"""
logger.info("## Saving and loading")

db.save_local("faiss_index")

new_db = FAISS.load_local("faiss_index", embeddings, asynchronous=True)

docs = await new_db.asimilarity_search(query)
logger.success(format_json(docs))

docs[0]

"""
# Serializing and De-Serializing to bytes

you can pickle the FAISS Index by these functions. If you use embeddings model which is of 90 mb (sentence-transformers/all-MiniLM-L6-v2 or any other model), the resultant pickle size would be more than 90 mb. the size of the model is also included in the overall size. To overcome this, use the below functions. These functions only serializes FAISS index and size would be much lesser. this can be helpful if you wish to store the index in database like sql.
"""
logger.info("# Serializing and De-Serializing to bytes")


pkl = db.serialize_to_bytes()  # serializes the faiss index
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.deserialize_from_bytes(
    embeddings=embeddings, serialized=pkl, asynchronous=True
)  # Load the index

"""
## Merging
You can also merge two FAISS vectorstores
"""
logger.info("## Merging")

db1 = await FAISS.afrom_texts(["foo"], embeddings)
logger.success(format_json(db1))
db2 = await FAISS.afrom_texts(["bar"], embeddings)
logger.success(format_json(db2))

db1.docstore._dict

db2.docstore._dict

db1.merge_from(db2)

db1.docstore._dict

"""
## Similarity Search with filtering
FAISS vectorstore can also support filtering, since the FAISS does not natively support filtering we have to do it manually. This is done by first fetching more results than `k` and then filtering them. You can filter the documents based on metadata. You can also set the `fetch_k` parameter when calling any search method to set how many documents you want to fetch before filtering. Here is a small example:
"""
logger.info("## Similarity Search with filtering")


list_of_documents = [
    Document(page_content="foo", metadata=dict(page=1)),
    Document(page_content="bar", metadata=dict(page=1)),
    Document(page_content="foo", metadata=dict(page=2)),
    Document(page_content="barbar", metadata=dict(page=2)),
    Document(page_content="foo", metadata=dict(page=3)),
    Document(page_content="bar burr", metadata=dict(page=3)),
    Document(page_content="foo", metadata=dict(page=4)),
    Document(page_content="bar bruh", metadata=dict(page=4)),
]
db = FAISS.from_documents(list_of_documents, embeddings)
results_with_scores = db.similarity_search_with_score("foo")
for doc, score in results_with_scores:
    logger.debug(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

"""
Now we make the same query call but we filter for only `page = 1`
"""
logger.info("Now we make the same query call but we filter for only `page = 1`")

results_with_scores = await db.asimilarity_search_with_score("foo", filter=dict(page=1))
logger.success(format_json(results_with_scores))
for doc, score in results_with_scores:
    logger.debug(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

"""
Same thing can be done with the `max_marginal_relevance_search` as well.
"""
logger.info("Same thing can be done with the `max_marginal_relevance_search` as well.")

results = await db.amax_marginal_relevance_search("foo", filter=dict(page=1))
logger.success(format_json(results))
for doc in results:
    logger.debug(f"Content: {doc.page_content}, Metadata: {doc.metadata}")

"""
Here is an example of how to set `fetch_k` parameter when calling `similarity_search`. Usually you would want the `fetch_k` parameter >> `k` parameter. This is because the `fetch_k` parameter is the number of documents that will be fetched before filtering. If you set `fetch_k` to a low number, you might not get enough documents to filter from.
"""
logger.info("Here is an example of how to set `fetch_k` parameter when calling `similarity_search`. Usually you would want the `fetch_k` parameter >> `k` parameter. This is because the `fetch_k` parameter is the number of documents that will be fetched before filtering. If you set `fetch_k` to a low number, you might not get enough documents to filter from.")

results = await db.asimilarity_search("foo", filter=dict(page=1), k=1, fetch_k=4)
logger.success(format_json(results))
for doc in results:
    logger.debug(f"Content: {doc.page_content}, Metadata: {doc.metadata}")

"""
Some [MongoDB query and projection operators](https://www.mongodb.com/docs/manual/reference/operator/query/) are supported for more advanced metadata filtering. The current list of supported operators are as follows:
- `$eq` (equals)
- `$neq` (not equals)
- `$gt` (greater than)
- `$lt` (less than)
- `$gte` (greater than or equal)
- `$lte` (less than or equal)
- `$in` (membership in list)
- `$nin` (not in list)
- `$and` (all conditions must match)
- `$or` (any condition must match)
- `$not` (negation of condition)

Performing the same above similarity search with advanced metadata filtering can be done as follows:
"""
logger.info("Some [MongoDB query and projection operators](https://www.mongodb.com/docs/manual/reference/operator/query/) are supported for more advanced metadata filtering. The current list of supported operators are as follows:")

results = await db.asimilarity_search(
        "foo", filter={"page": {"$eq": 1}}, k=1, fetch_k=4
    )
logger.success(format_json(results))
for doc in results:
    logger.debug(f"Content: {doc.page_content}, Metadata: {doc.metadata}")

"""
## Delete

You can also delete ids. Note that the ids to delete should be the ids in the docstore.
"""
logger.info("## Delete")

db.delete([db.index_to_docstore_id[0]])

0 in db.index_to_docstore_id

logger.info("\n\n[DONE]", bright=True)