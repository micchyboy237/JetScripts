from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_deeplake.vectorstores import DeeplakeVectorStore
from langchain_text_splitters import CharacterTextSplitter
import deeplake
import os
import random
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
#  Activeloop Deep Lake

>[Activeloop Deep Lake](https://docs.deeplake.ai/) as a Multi-Modal Vector Store that stores embeddings and their metadata including text, jsons, images, audio, video, and more. It saves the data locally, in your cloud, or on Activeloop storage. It performs hybrid search including embeddings and their attributes.

This notebook showcases basic functionality related to `Activeloop Deep Lake`. While `Deep Lake` can store embeddings, it is capable of storing any type of data. It is a serverless data lake with version control, query engine and streaming dataloaders to deep learning frameworks.  

For more information, please see the Deep Lake [documentation](https://docs.deeplake.ai/)

## Setting up
"""
logger.info("#  Activeloop Deep Lake")

# %pip install --upgrade --quiet  langchain-ollama langchain-deeplake tiktoken

"""
## Example provided by Activeloop

[Integration with LangChain](https://docs.activeloop.ai/tutorials/vector-store/deep-lake-vector-store-in-langchain).

## Deep Lake locally
"""
logger.info("## Example provided by Activeloop")


# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")

if "ACTIVELOOP_TOKEN" not in os.environ:
#     os.environ["ACTIVELOOP_TOKEN"] = getpass.getpass("activeloop token:")


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

"""
### Create a local dataset

Create a dataset locally at `./my_deeplake/`, then run similarity search. The Deeplake+LangChain integration uses Deep Lake datasets under the hood, so `dataset` and `vector store` are used interchangeably. To create a dataset in your own cloud, or in the Deep Lake storage, [adjust the path accordingly](https://docs.deeplake.ai/latest/getting-started/storage-and-creds/storage-options/).
"""
logger.info("### Create a local dataset")

db = DeeplakeVectorStore(
    dataset_path="./my_deeplake/", embedding_function=embeddings, overwrite=True
)
db.add_documents(docs)

"""
### Query dataset
"""
logger.info("### Query dataset")

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

logger.debug(docs[0].page_content)

"""
Later, you can reload the dataset without recomputing embeddings
"""
logger.info("Later, you can reload the dataset without recomputing embeddings")

db = DeeplakeVectorStore(
    dataset_path="./my_deeplake/", embedding_function=embeddings, read_only=True
)
docs = db.similarity_search(query)

"""
Setting `read_only=True` revents accidental modifications to the vector store when updates are not needed. This ensures that the data remains unchanged unless explicitly intended. It is generally a good practice to specify this argument to avoid unintended updates.

### Retrieval Question/Answering
"""
logger.info("### Retrieval Question/Answering")


qa = RetrievalQA.from_chain_type(
    llm=ChatOllama(model="llama3.2"),
    chain_type="stuff",
    retriever=db.as_retriever(),
)

query = "What did the president say about Ketanji Brown Jackson"
qa.run(query)

"""
### Attribute based filtering in metadata

Let's create another vector store containing metadata with the year the documents were created.
"""
logger.info("### Attribute based filtering in metadata")


for d in docs:
    d.metadata["year"] = random.randint(2012, 2014)

db = DeeplakeVectorStore.from_documents(
    docs, embeddings, dataset_path="./my_deeplake/", overwrite=True
)

db.similarity_search(
    "What did the president say about Ketanji Brown Jackson",
    filter={"metadata": {"year": 2013}},
)

"""
### Choosing distance function
Distance function `L2` for Euclidean, `cos` for cosine similarity
"""
logger.info("### Choosing distance function")

db.similarity_search(
    "What did the president say about Ketanji Brown Jackson?", distance_metric="l2"
)

"""
### Maximal Marginal relevance
Using maximal marginal relevance
"""
logger.info("### Maximal Marginal relevance")

db.max_marginal_relevance_search(
    "What did the president say about Ketanji Brown Jackson?"
)

"""
### Delete dataset
"""
logger.info("### Delete dataset")

db.delete_dataset()

"""
## Deep Lake datasets on cloud (Activeloop, AWS, GCS, etc.) or in memory
By default, Deep Lake datasets are stored locally. To store them in memory, in the Deep Lake Managed DB, or in any object storage, you can provide the [corresponding path and credentials when creating the vector store](https://docs.deeplake.ai/latest/getting-started/storage-and-creds/storage-options/). Some paths require registration with Activeloop and creation of an API token that can be [retrieved here](https://app.activeloop.ai/)
"""
logger.info("## Deep Lake datasets on cloud (Activeloop, AWS, GCS, etc.) or in memory")

os.environ["ACTIVELOOP_TOKEN"] = activeloop_token

username = "<USERNAME_OR_ORG>"  # your username on app.activeloop.ai
dataset_path = f"hub://{username}/langchain_testing_python"  # could be also ./local/path (much faster locally), s3://bucket/path/to/dataset, gcs://path/to/dataset, etc.

docs = text_splitter.split_documents(documents)

embedding = OllamaEmbeddings(model="mxbai-embed-large")
db = DeeplakeVectorStore(
    dataset_path=dataset_path, embedding_function=embeddings, overwrite=True
)
ids = db.add_documents(docs)

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
logger.debug(docs[0].page_content)

username = "<USERNAME_OR_ORG>"  # your username on app.activeloop.ai
dataset_path = f"hub://{username}/langchain_testing"

docs = text_splitter.split_documents(documents)

embedding = OllamaEmbeddings(model="mxbai-embed-large")
db = DeeplakeVectorStore(
    dataset_path=dataset_path,
    embedding_function=embeddings,
    overwrite=True,
)
ids = db.add_documents(docs)

"""
### TQL Search

Furthermore, the execution of queries is also supported within the similarity_search method, whereby the query can be specified utilizing Deep Lake's Tensor Query Language (TQL).
"""
logger.info("### TQL Search")

search_id = db.dataset["ids"][0]

docs = db.similarity_search(
    query=None,
    tql=f"SELECT * WHERE ids == '{search_id}'",
)

db.dataset.summary()

"""
### Creating vector stores on AWS S3
"""
logger.info("### Creating vector stores on AWS S3")

dataset_path = "s3://BUCKET/langchain_test"  # could be also ./local/path (much faster locally), hub://bucket/path/to/dataset, gcs://path/to/dataset, etc.

embedding = OllamaEmbeddings(model="mxbai-embed-large")
db = DeeplakeVectorStore.from_documents(
    docs,
    dataset_path=dataset_path,
    embedding=embeddings,
    overwrite=True,
    creds={
        "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "aws_session_token": os.environ["AWS_SESSION_TOKEN"],  # Optional
    },
)

"""
## Deep Lake API
you can access the Deep Lake  dataset at `db.vectorstore`
"""
logger.info("## Deep Lake API")

db.dataset.summary()

embeds = db.dataset["embeddings"][:]

"""
### Transfer local dataset to cloud
Copy already created dataset to the cloud. You can also transfer from cloud to local.
"""
logger.info("### Transfer local dataset to cloud")


username = "<USERNAME_OR_ORG>"  # your username on app.activeloop.ai
source = f"hub://{username}/langchain_testing"  # could be local, s3, gcs, etc.
destination = f"hub://{username}/langchain_test_copy"  # could be local, s3, gcs, etc.


deeplake.copy(src=source, dst=destination)

db = DeeplakeVectorStore(dataset_path=destination, embedding_function=embeddings)
db.add_documents(docs)

logger.info("\n\n[DONE]", bright=True)