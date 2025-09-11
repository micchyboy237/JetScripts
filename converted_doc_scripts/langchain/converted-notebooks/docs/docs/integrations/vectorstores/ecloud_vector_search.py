from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import EcloudESVectorStore
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
# China Mobile ECloud ElasticSearch VectorSearch

>[China Mobile ECloud VectorSearch](https://ecloud.10086.cn/portal/product/elasticsearch) is a fully managed, enterprise-level distributed search and analysis service. China Mobile ECloud VectorSearch provides low-cost, high-performance, and reliable retrieval and analysis platform level product services for structured/unstructured data. As a vector database , it supports multiple index types and similarity distance methods. 

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

This notebook shows how to use functionality related to the `ECloud ElasticSearch VectorStore`.
To run, you should have an [China Mobile ECloud VectorSearch](https://ecloud.10086.cn/portal/product/elasticsearch) instance up and running:

Read the [help document](https://ecloud.10086.cn/op-help-center/doc/category/1094) to quickly familiarize and configure China Mobile ECloud ElasticSearch instance.

After the instance is up and running, follow these steps to split documents, get embeddings, connect to the baidu cloud elasticsearch instance, index documents, and perform vector retrieval.
"""
logger.info("# China Mobile ECloud ElasticSearch VectorSearch")


"""
First, we want to use `OllamaEmbeddings` so we have to get the Ollama API Key.
"""
logger.info(
    "First, we want to use `OllamaEmbeddings` so we have to get the Ollama API Key.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")

"""
Secondly, split documents and get embeddings.
"""
logger.info("Secondly, split documents and get embeddings.")


loader = TextLoader("../../../state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

ES_URL = "http://localhost:9200"
USER = "your user name"
PASSWORD = "your password"
indexname = "your index name"

"""
then, index documents
"""
logger.info("then, index documents")

docsearch = EcloudESVectorStore.from_documents(
    docs,
    embeddings,
    es_url=ES_URL,
    user=USER,
    password=PASSWORD,
    index_name=indexname,
    refresh_indices=True,
)

"""
Finally, Query and retrieve data
"""
logger.info("Finally, Query and retrieve data")

query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query, k=10)
logger.debug(docs[0].page_content)

"""
A commonly used case
"""
logger.info("A commonly used case")


def test_dense_float_vectore_lsh_cosine() -> None:
    """
    Test indexing with vectore type knn_dense_float_vector and  model-similarity of lsh-cosine
    this mapping is compatible with model of exact and similarity of l2/cosine
    this mapping is compatible with model of lsh and similarity of cosine
    """
    docsearch = EcloudESVectorStore.from_documents(
        docs,
        embeddings,
        es_url=ES_URL,
        user=USER,
        password=PASSWORD,
        index_name=indexname,
        refresh_indices=True,
        text_field="my_text",
        vector_field="my_vec",
        vector_type="knn_dense_float_vector",
        vector_params={"model": "lsh",
                       "similarity": "cosine", "L": 99, "k": 1},
    )

    docs = docsearch.similarity_search(
        query,
        k=10,
        search_params={
            "model": "exact",
            "vector_field": "my_vec",
            "text_field": "my_text",
        },
    )
    logger.debug(docs[0].page_content)

    docs = docsearch.similarity_search(
        query,
        k=10,
        search_params={
            "model": "exact",
            "similarity": "l2",
            "vector_field": "my_vec",
            "text_field": "my_text",
        },
    )
    logger.debug(docs[0].page_content)

    docs = docsearch.similarity_search(
        query,
        k=10,
        search_params={
            "model": "exact",
            "similarity": "cosine",
            "vector_field": "my_vec",
            "text_field": "my_text",
        },
    )
    logger.debug(docs[0].page_content)

    docs = docsearch.similarity_search(
        query,
        k=10,
        search_params={
            "model": "lsh",
            "similarity": "cosine",
            "candidates": 10,
            "vector_field": "my_vec",
            "text_field": "my_text",
        },
    )
    logger.debug(docs[0].page_content)


"""
With filter case
"""
logger.info("With filter case")


def test_dense_float_vectore_exact_with_filter() -> None:
    """
    Test indexing with vectore type knn_dense_float_vector and default model/similarity
    this mapping is compatible with model of exact and similarity of l2/cosine
    """
    docsearch = EcloudESVectorStore.from_documents(
        docs,
        embeddings,
        es_url=ES_URL,
        user=USER,
        password=PASSWORD,
        index_name=indexname,
        refresh_indices=True,
        text_field="my_text",
        vector_field="my_vec",
        vector_type="knn_dense_float_vector",
    )
    docs = docsearch.similarity_search(
        query,
        k=10,
        filter={"match_all": {}},
        search_params={
            "model": "exact",
            "vector_field": "my_vec",
            "text_field": "my_text",
        },
    )
    logger.debug(docs[0].page_content)

    docs = docsearch.similarity_search(
        query,
        k=10,
        filter={"term": {"my_text": "Jackson"}},
        search_params={
            "model": "exact",
            "vector_field": "my_vec",
            "text_field": "my_text",
        },
    )
    logger.debug(docs[0].page_content)

    docs = docsearch.similarity_search(
        query,
        k=10,
        filter={"term": {"my_text": "president"}},
        search_params={
            "model": "exact",
            "similarity": "l2",
            "vector_field": "my_vec",
            "text_field": "my_text",
        },
    )
    logger.debug(docs[0].page_content)


logger.info("\n\n[DONE]", bright=True)
