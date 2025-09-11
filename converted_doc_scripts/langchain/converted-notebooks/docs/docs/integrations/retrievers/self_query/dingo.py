from dingodb import DingoDB
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Dingo
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
# DingoDB

>[DingoDB](https://dingodb.readthedocs.io/en/latest/) is a distributed multi-mode vector database, which combines the characteristics of data lakes and vector databases, and can store data of any type and size (Key-Value, PDF, audio, video, etc.). It has real-time low-latency processing capabilities to achieve rapid insight and response, and can efficiently conduct instant analysis and process multi-modal data.

In the walkthrough, we'll demo the `SelfQueryRetriever` with a `DingoDB` vector store.

## Creating a DingoDB index
First we'll want to create a `DingoDB` vector store and seed it with some data. We've created a small demo set of documents that contain summaries of movies.

To use DingoDB, you should have a [DingoDB instance up and running](https://github.com/dingodb/dingo-deploy/blob/main/README.md).

**Note:** The self-query retriever requires you to have `lark` package installed.
"""
logger.info("# DingoDB")

# %pip install --upgrade --quiet  dingodb
# %pip install --upgrade --quiet  git+https://git@github.com/dingodb/pydingo.git

"""
We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.
"""
logger.info(
    "We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.")


# OPENAI_API_KEY = ""

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

"""

"""


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

index_name = "langchain_demo"

dingo_client = DingoDB(user="", password="", host=["172.30.14.221:13000"])
if (
    index_name not in dingo_client.get_index()
    and index_name.upper() not in dingo_client.get_index()
):
    dingo_client.create_index(
        index_name=index_name, dimension=1536, metric_type="cosine", auto_id=False
    )

docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7,
                  "genre": '"action", "science fiction"'},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": '"science fiction", "thriller"',
            "rating": 9.9,
        },
    ),
]
vectorstore = Dingo.from_documents(
    docs, embeddings, index_name=index_name, client=dingo_client
)

dingo_client.get_index()
dingo_client.delete_index("langchain_demo")

dingo_client.vector_count("langchain_demo")

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

retriever.invoke("I want to watch a movie rated higher than 8.5")

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

retriever.invoke("What are two movies about dinosaurs")

logger.info("\n\n[DONE]", bright=True)
