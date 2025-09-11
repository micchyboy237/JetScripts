from jet.models.config import MODELS_CACHE_DIR
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import SemaDB
from langchain_community.vectorstores.utils import DistanceStrategy
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
# SemaDB

> [SemaDB](https://www.semafind.com/products/semadb) from [SemaFind](https://www.semafind.com) is a no fuss vector similarity database for building AI applications. The hosted `SemaDB Cloud` offers a no fuss developer experience to get started.

The full documentation of the API along with examples and an interactive playground is available on [RapidAPI](https://rapidapi.com/semafind-semadb/api/semadb).

This notebook demonstrates usage of the `SemaDB Cloud` vector store.

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

## Load document embeddings

To run things locally, we are using [Sentence Transformers](https://www.sbert.net/) which are commonly used for embedding sentences. You can use any embedding model LangChain offers.
"""
logger.info("# SemaDB")

# %pip install --upgrade --quiet  sentence_transformers


model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
logger.debug(len(docs))

"""
## Connect to SemaDB

SemaDB Cloud uses [RapidAPI keys](https://rapidapi.com/semafind-semadb/api/semadb) to authenticate. You can obtain yours by creating a free RapidAPI account.
"""
logger.info("## Connect to SemaDB")

# import getpass

if "SEMADB_API_KEY" not in os.environ:
#     os.environ["SEMADB_API_KEY"] = getpass.getpass("SemaDB API Key:")


"""
The parameters to the SemaDB vector store reflect the API directly:

- "mycollection": is the collection name in which we will store these vectors.
- 768: is dimensions of the vectors. In our case, the sentence transformer embeddings yield 768 dimensional vectors.
- API_KEY: is your RapidAPI key.
- embeddings: correspond to how the embeddings of documents, texts and queries will be generated.
- DistanceStrategy: is the distance metric used. The wrapper automatically normalises vectors if COSINE is used.
"""
logger.info("The parameters to the SemaDB vector store reflect the API directly:")

db = SemaDB("mycollection", 768, embeddings, DistanceStrategy.COSINE)

db.create_collection()

"""
The SemaDB vector store wrapper adds the document text as point metadata to collect later. Storing large chunks of text is *not recommended*. If you are indexing a large collection, we instead recommend storing references to the documents such as external Ids.
"""
logger.info("The SemaDB vector store wrapper adds the document text as point metadata to collect later. Storing large chunks of text is *not recommended*. If you are indexing a large collection, we instead recommend storing references to the documents such as external Ids.")

db.add_documents(docs)[:2]

"""
## Similarity Search

We use the default LangChain similarity search interface to search for the most similar sentences.
"""
logger.info("## Similarity Search")

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
logger.debug(docs[0].page_content)

docs = db.similarity_search_with_score(query)
docs[0]

"""
## Clean up

You can delete the collection to remove all data.
"""
logger.info("## Clean up")

db.delete_collection()

logger.info("\n\n[DONE]", bright=True)