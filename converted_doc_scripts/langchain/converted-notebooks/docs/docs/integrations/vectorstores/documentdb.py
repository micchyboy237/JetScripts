from jet.adapters.langchain.chat_ollama import Ollama
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain.chains import RetrievalQA
from langchain.vectorstores.documentdb import (
DocumentDBSimilarityType,
DocumentDBVectorSearch,
)
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from pymongo import MongoClient
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
# Amazon Document DB

>[Amazon DocumentDB (with MongoDB Compatibility)](https://docs.aws.amazon.com/documentdb/) makes it easy to set up, operate, and scale MongoDB-compatible databases in the cloud.
> With Amazon DocumentDB, you can run the same application code and use the same drivers and tools that you use with MongoDB.
> Vector search for Amazon DocumentDB combines the flexibility and rich querying capability of a JSON-based document database with the power of vector search.


This notebook shows you how to use [Amazon Document DB Vector Search](https://docs.aws.amazon.com/documentdb/latest/developerguide/vector-search.html) to store documents in collections, create indicies and perform vector search queries using approximate nearest neighbor algorithms such "cosine", "euclidean", and "dotProduct". By default, DocumentDB creates Hierarchical Navigable Small World (HNSW) indexes. To learn about other supported vector index types, please refer to the document linked above.

To use DocumentDB, you must first deploy a cluster. Please refer to the [Developer Guide](https://docs.aws.amazon.com/documentdb/latest/developerguide/what-is.html) for more details.

[Sign Up](https://aws.amazon.com/free/) for free to get started today.
"""
logger.info("# Amazon Document DB")

# !pip install pymongo

# import getpass

# CONNECTION_STRING = getpass.getpass("DocumentDB Cluster URI:")

INDEX_NAME = "izzy-test-index"
NAMESPACE = "izzy_test_db.izzy_test_collection"
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")

"""
We want to use `OllamaEmbeddings` so we need to set up our Ollama environment variables.
"""
logger.info("We want to use `OllamaEmbeddings` so we need to set up our Ollama environment variables.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")
os.environ["OPENAI_EMBEDDINGS_DEPLOYMENT"] = (
    "smart-agent-embedding-ada"  # the deployment name for the embedding model
)
os.environ["OPENAI_EMBEDDINGS_MODEL_NAME"] = "text-embedding-ada-002"  # the model name

"""
Now, we will load the documents into the collection, create the index, and then perform queries against the index.

Please refer to the [documentation](https://docs.aws.amazon.com/documentdb/latest/developerguide/vector-search.html) if you have questions about certain parameters
"""
logger.info("Now, we will load the documents into the collection, create the index, and then perform queries against the index.")


SOURCE_FILE_NAME = "../../how_to/state_of_the_union.txt"

loader = TextLoader(SOURCE_FILE_NAME)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

model_deployment = os.getenv(
    "OPENAI_EMBEDDINGS_DEPLOYMENT", "smart-agent-embedding-ada"
)
model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002")


ollama_embeddings: OllamaEmbeddings = OllamaEmbeddings(
    deployment=model_deployment, model=model_name
)


INDEX_NAME = "izzy-test-index-2"
NAMESPACE = "izzy_test_db.izzy_test_collection"
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")

client: MongoClient = MongoClient(CONNECTION_STRING)
collection = client[DB_NAME][COLLECTION_NAME]

model_deployment = os.getenv(
    "OPENAI_EMBEDDINGS_DEPLOYMENT", "smart-agent-embedding-ada"
)
model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002")

vectorstore = DocumentDBVectorSearch.from_documents(
    documents=docs,
    embedding=ollama_embeddings,
    collection=collection,
    index_name=INDEX_NAME,
)

dimensions = 1536

similarity_algorithm = DocumentDBSimilarityType.COS

vectorstore.create_index(dimensions, similarity_algorithm)

query = "What did the President say about Ketanji Brown Jackson"
docs = vectorstore.similarity_search(query)

logger.debug(docs[0].page_content)

"""
Once the documents have been loaded and the index has been created, you can now instantiate the vector store directly and run queries against the index
"""
logger.info("Once the documents have been loaded and the index has been created, you can now instantiate the vector store directly and run queries against the index")

vectorstore = DocumentDBVectorSearch.from_connection_string(
    connection_string=CONNECTION_STRING,
    namespace=NAMESPACE,
    embedding=ollama_embeddings,
    index_name=INDEX_NAME,
)

query = "What did the president say about Ketanji Brown Jackson"
docs = vectorstore.similarity_search(query)

logger.debug(docs[0].page_content)

query = "Which stats did the President share about the U.S. economy"
docs = vectorstore.similarity_search(query)

logger.debug(docs[0].page_content)

"""
## Question Answering
"""
logger.info("## Question Answering")

qa_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 25},
)


prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


qa = RetrievalQA.from_chain_type(
    llm=Ollama(),
    chain_type="stuff",
    retriever=qa_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

docs = qa({"query": "gpt-4 compute requirements"})

logger.debug(docs["result"])
logger.debug(docs["source_documents"])

logger.info("\n\n[DONE]", bright=True)