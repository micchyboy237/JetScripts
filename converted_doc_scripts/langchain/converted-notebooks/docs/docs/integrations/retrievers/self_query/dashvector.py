from jet.logger import logger
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain_community.vectorstores import DashVector
from langchain_core.documents import Document
import dashvector
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
# DashVector

> [DashVector](https://help.aliyun.com/document_detail/2510225.html) is a fully managed vector DB service that supports high-dimension dense and sparse vectors, real-time insertion and filtered search. It is built to scale automatically and can adapt to different application requirements.
> The vector retrieval service `DashVector` is based on the `Proxima` core of the efficient vector engine independently developed by `DAMO Academy`,
>  and provides a cloud-native, fully managed vector retrieval service with horizontal expansion capabilities.
>  `DashVector` exposes its powerful vector management, vector query and other diversified capabilities through a simple and
> easy-to-use SDK/API interface, which can be quickly integrated by upper-layer AI applications, thereby providing services
> including large model ecology, multi-modal AI search, molecular structure A variety of application scenarios, including analysis,
> provide the required efficient vector retrieval capabilities.

In this notebook, we'll demo the `SelfQueryRetriever` with a `DashVector` vector store.

## Create DashVector vectorstore

First we'll want to create a `DashVector` VectorStore and seed it with some data. We've created a small demo set of documents that contain summaries of movies.

To use DashVector, you have to have `dashvector` package installed, and you must have an API key and an Environment. Here are the [installation instructions](https://help.aliyun.com/document_detail/2510223.html).

NOTE: The self-query retriever requires you to have `lark` package installed.
"""
logger.info("# DashVector")

# %pip install --upgrade --quiet  lark dashvector



client = dashvector.Client(api_key=os.environ["DASHVECTOR_API_KEY"])


embeddings = DashScopeEmbeddings()

client.create("langchain-self-retriever-demo", dimension=1536)

docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "action"},
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
            "genre": "science fiction",
            "rating": 9.9,
        },
    ),
]
vectorstore = DashVector.from_documents(
    docs, embeddings, collection_name="langchain-self-retriever-demo"
)

"""
## Create your self-querying retriever

Now we can instantiate our retriever. To do this we'll need to provide some information upfront about the metadata fields that our documents support and a short description of the document contents.
"""
logger.info("## Create your self-querying retriever")


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
llm = Tongyi(temperature=0)
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