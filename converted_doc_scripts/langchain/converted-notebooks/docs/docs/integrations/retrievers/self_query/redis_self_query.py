from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Redis
from langchain_core.documents import Document
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
# Redis

>[Redis](https://redis.com) is an open-source key-value store that can be used as a cache, message broker, database, vector database and more.

In the notebook, we'll demo the `SelfQueryRetriever` wrapped around a `Redis` vector store.

## Creating a Redis vector store
First we'll want to create a Redis vector store and seed it with some data. We've created a small demo set of documents that contain summaries of movies.

**Note:** The self-query retriever requires you to have `lark` installed (`pip install lark`) along with integration-specific requirements.
"""
logger.info("# Redis")

# %pip install --upgrade --quiet  redis redisvl langchain-ollama tiktoken lark

"""
We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.
"""
logger.info(
    "We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={
            "year": 1993,
            "rating": 7.7,
            "director": "Steven Spielberg",
            "genre": "science fiction",
        },
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={
            "year": 2010,
            "director": "Christopher Nolan",
            "genre": "science fiction",
            "rating": 8.2,
        },
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={
            "year": 2006,
            "director": "Satoshi Kon",
            "genre": "science fiction",
            "rating": 8.6,
        },
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={
            "year": 2019,
            "director": "Greta Gerwig",
            "genre": "drama",
            "rating": 8.3,
        },
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={
            "year": 1995,
            "director": "John Lasseter",
            "genre": "animated",
            "rating": 9.1,
        },
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "rating": 9.9,
            "director": "Andrei Tarkovsky",
            "genre": "science fiction",
        },
    ),
]

index_schema = {
    "tag": [{"name": "genre"}],
    "text": [{"name": "director"}],
    "numeric": [{"name": "year"}, {"name": "rating"}],
}

vectorstore = Redis.from_documents(
    docs,
    embeddings,
    redis_url="redis://localhost:6379",
    index_name="movie_reviews",
    index_schema=index_schema,
)

"""
## Creating our self-querying retriever
Now we can instantiate our retriever. To do this we'll need to provide some information upfront about the metadata fields that our documents support and a short description of the document contents.
"""
logger.info("## Creating our self-querying retriever")


metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
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

retriever.invoke("I want to watch a movie rated higher than 8.4")

retriever.invoke("Has Greta Gerwig directed any movies about women")

retriever.invoke("What's a highly rated (above 8.5) science fiction film?")

retriever.invoke(
    "What's a movie after 1990 but before 2005 that's all about toys, and preferably is animated"
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
    enable_limit=True,
    verbose=True,
)

retriever.invoke("what are two movies about dinosaurs")

logger.info("\n\n[DONE]", bright=True)
