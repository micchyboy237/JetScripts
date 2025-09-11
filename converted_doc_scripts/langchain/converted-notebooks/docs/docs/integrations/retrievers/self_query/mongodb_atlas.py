from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
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
# MongoDB Atlas

>[MongoDB Atlas](https://www.mongodb.com/) is a document database that can be 
used as a vector database.

In the walkthrough, we'll demo the `SelfQueryRetriever` with a `MongoDB Atlas` vector store.

## Creating a MongoDB Atlas vectorstore
First we'll want to create a MongoDB Atlas VectorStore and seed it with some data. We've created a small demo set of documents that contain summaries of movies.

NOTE: The self-query retriever requires you to have `lark` installed (`pip install lark`). We also need the `pymongo` package.
"""
logger.info("# MongoDB Atlas")

# %pip install --upgrade --quiet  lark pymongo

"""
We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.
"""
logger.info(
    "We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.")


# OPENAI_API_KEY = "Use your Ollama key"

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


CONNECTION_STRING = "Use your MongoDB Atlas connection string"
DB_NAME = "Name of your MongoDB Atlas database"
COLLECTION_NAME = "Name of your collection in the database"
INDEX_NAME = "Name of a search index defined on the collection"

MongoClient = MongoClient(CONNECTION_STRING)
collection = MongoClient[DB_NAME][COLLECTION_NAME]

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "action"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "genre": "thriller", "rating": 8.2},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "rating": 8.3, "genre": "drama"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={"year": 1979, "rating": 9.9, "genre": "science fiction"},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "genre": "thriller", "rating": 9.0},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated", "rating": 9.3},
    ),
]

vectorstore = MongoDBAtlasVectorSearch.from_documents(
    docs,
    embeddings,
    collection=collection,
    index_name=INDEX_NAME,
)

"""
Now, let's create a vector search index on your cluster. In the below example, `embedding` is the name of the field that contains the embedding vector. Please refer to the [documentation](https://www.mongodb.com/docs/atlas/atlas-search/field-types/knn-vector) to get more details on how to define an Atlas Vector Search index.
You can name the index `{COLLECTION_NAME}` and create the index on the namespace `{DB_NAME}.{COLLECTION_NAME}`. Finally, write the following definition in the JSON editor on MongoDB Atlas:

```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "dimensions": 1536,
        "similarity": "cosine",
        "type": "knnVector"
      },
      "genre": {
        "type": "token"
      },
      "ratings": {
        "type": "number"
      },
      "year": {
        "type": "number"
      }
    }
  }
}
```

## Creating our self-querying retriever
Now we can instantiate our retriever. To do this we'll need to provide some information upfront about the metadata fields that our documents support and a short description of the document contents.
"""
logger.info("## Creating our self-querying retriever")


metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]
document_content_description = "Brief summary of a movie"

llm = ChatOllama(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)

"""
## Testing it out
And now we can try actually using our retriever!
"""
logger.info("## Testing it out")

retriever.invoke("What are some movies about dinosaurs")

retriever.invoke("What are some highly rated movies (above 9)?")

retriever.invoke("I want to watch a movie about toys rated higher than 9")

retriever.invoke("What's a highly rated (above or equal 9) thriller film?")

retriever.invoke(
    "What's a movie after 1990 but before 2005 that's all about dinosaurs, \
    and preferably has a lot of action"
)

"""
## Filter k

We can also use the self query retriever to specify `k`: the number of documents to fetch.

We can do this by passing `enable_limit=True` to the constructor.
"""
logger.info("## Filter k")

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
    enable_limit=True,
)

retriever.invoke("What are two movies about dinosaurs?")

logger.info("\n\n[DONE]", bright=True)
