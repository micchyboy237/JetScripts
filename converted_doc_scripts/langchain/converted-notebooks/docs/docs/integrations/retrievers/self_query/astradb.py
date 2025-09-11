from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
   OllamaEmbeddings(model="nomic-embed-text"))
        OllamaEmbeddings(model="nomic-embed-text"))
        from jet.logger import logger
        from langchain.chains.query_constructor.schema import AttributeInfo
        from langchain.retrievers.self_query.base import SelfQueryRetriever
        from langchain_astradb import AstraDBVectorStore
        from langchain_core.documents import Document
        import os
        import shutil


        OUTPUT_DIR= os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        log_file= os.path.join(OUTPUT_DIR, "main.log")
        logger.basicConfig(filename=log_file)
        logger.info(f"Logs: {log_file}")

        PERSIST_DIR= f"{OUTPUT_DIR}/chroma"
        os.makedirs(PERSIST_DIR, exist_ok=True)

        """
# Astra DB

> [DataStax Astra DB](https://docs.datastax.com/en/astra-db-serverless/index.html) is a serverless
> AI-ready database built on `Apache Cassandra®` and made conveniently available
> through an easy-to-use JSON API.

In the walkthrough, we'll demo the `SelfQueryRetriever` with an `Astra DB` vector store.

## Creating an Astra DB vector store
First, create an Astra DB vector store and seed it with some data.

We've created a small demo set of documents containing movie summaries.

NOTE: The self-query retriever requires the `lark` package installed (`pip install lark`).
"""
        logger.info("# Astra DB")

        # !pip install "langchain-astradb>=0.6,<0.7"
        "jet.adapters.langchain.chat_ollama>=0.3,<0.4"
        "lark>=1.2,<2.0"

        """
In this example, you'll use the `OllamaEmbeddings`. Please enter an Ollama API Key.
"""
        logger.info(
    "In this example, you'll use the `OllamaEmbeddings`. Please enter an Ollama API Key.")

        # from getpass import getpass


        # if "OPENAI_API_KEY" not in os.environ:
        #     os.environ["OPENAI_API_KEY"] = getpass("Ollama API Key:")

        embeddings= OllamaEmbeddings(model="nomic-embed-text")

        """
Create the Astra DB VectorStore:

- the API Endpoint looks like `https://01234567-89ab-cdef-0123-456789abcdef-us-east1.apps.astra.datastax.com`
- the Token looks like `AstraCS:aBcD0123...`
"""
        logger.info("Create the Astra DB VectorStore:")

        ASTRA_DB_API_ENDPOINT= input("ASTRA_DB_API_ENDPOINT = ")
        # ASTRA_DB_APPLICATION_TOKEN = getpass("ASTRA_DB_APPLICATION_TOKEN = ")


        docs= [
    Document(
        page_content = "A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata = {"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
        Document(
        page_content = "Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata = {"year": 2010,
            "director": "Christopher Nolan", "rating": 8.2},
    ),
        Document(
        page_content = "A psychologist / detective gets lost in a series of dreams within dreams "
        "within dreams and Inception reused the idea",
        metadata = {"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
        Document(
        page_content = "A bunch of normal-sized women are supremely wholesome and some men "
        "pine after them",
        metadata = {"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
        Document(
        page_content = "Toys come alive and have a blast doing so",
        metadata = {"year": 1995, "genre": "animated"},
    ),
        Document(
        page_content = "Three men walk into the Zone, three men walk out of the Zone",
        metadata = {
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "science fiction",
            "rating": 9.9,
        },
    ),
    ]

    vectorstore = AstraDBVectorStore.from_documents(
    docs,
    embeddings,
    collection_name="astra_self_query_demo",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)

        """
## Creating a self-querying retriever

Now you can instantiate the retriever.

To do this, you need to provide some information upfront about the metadata fields that the documents support, along with a short description of the documents' contents.
"""
        logger.info("## Creating a self-querying retriever")


        metadata_field_info = [
    AttributeInfo(
        name = "genre",
        description = "The genre of the movie",
        type = "string or list[string]",
    ),
        AttributeInfo(
        name = "year",
        description = "The year the movie was released",
        type = "integer",
    ),
        AttributeInfo(
        name = "director",
        description = "The name of the movie director",
        type = "string",
    ),
        AttributeInfo(
        name ="rating", description="A 1-10 rating for the movie", type="float"
    ),
    ]
    document_content_description = "Brief summary of a movie"
    llm = ChatOllama(temperature=0)

    retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
)

        """
## Testing it out

Now you can try actually using our retriever:
"""
        logger.info("## Testing it out")

        retriever.invoke("What are some movies about dinosaurs?")

        retriever.invoke("I want to watch a movie rated higher than 8.5")

        retriever.invoke("Has Greta Gerwig directed any movies about women")

        retriever.invoke("What's a highly rated (above 8.5), science fiction movie ?")

        retriever.invoke(
    "What's a movie about toys after 1990 but before 2005, and is animated"
)

        """
## Set a limit ('k')

you can also use the self-query retriever to specify `k`, the number of documents to fetch.

You achieve this by passing `enable_limit=True` to the constructor.
"""
        logger.info("## Set a limit ('k')")

        retriever_k = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
    enable_limit=True,
)

        retriever_k.invoke("What are two movies about dinosaurs?")

        """
## Cleanup

If you want to completely delete the collection from your Astra DB instance, run this.

_(You will lose the data you stored in it.)_
"""
        logger.info("## Cleanup")

        vectorstore.delete_collection()

        logger.info("\n\n[DONE]", bright=True)
