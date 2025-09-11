from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma import Chroma
from langchain_community.query_constructors.chroma import ChromaTranslator
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
# How to do "self-querying" retrieval

:::info

Head to [Integrations](/docs/integrations/retrievers/self_query) for documentation on vector stores with built-in support for self-querying.

:::

A self-querying [retriever](/docs/concepts/retrievers/) is one that, as the name suggests, has the ability to query itself. Specifically, given any natural language query, the retriever uses a query-constructing LLM chain to write a structured query and then applies that structured query to its underlying [vector store](/docs/concepts/vectorstores/). This allows the retriever to not only use the user-input query for semantic similarity comparison with the contents of stored documents but to also extract filters from the user query on the metadata of stored documents and to execute those filters.

![](../../static/img/self_querying.jpg)

## Get started
For demonstration purposes we'll use a `Chroma` vector store. We've created a small demo set of documents that contain summaries of movies.

**Note:** The self-query retriever requires you to have `lark` package installed.
"""
logger.info("# How to do "self-querying" retrieval")

# %pip install --upgrade --quiet  lark langchain-chroma


docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
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
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]
vectorstore = Chroma.from_documents(
    docs, OllamaEmbeddings(model="nomic-embed-text"))

"""
### Creating our self-querying retriever

Now we can instantiate our retriever. To do this we'll need to provide some information upfront about the metadata fields that our documents support and a short description of the document contents.
"""
logger.info("### Creating our self-querying retriever")


metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        type="string",
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
llm = ChatOllama(model="llama3.2")
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
)

"""
### Testing it out

And now we can actually try using our retriever!
"""
logger.info("### Testing it out")

retriever.invoke("I want to watch a movie rated higher than 8.5")

retriever.invoke("Has Greta Gerwig directed any movies about women")

retriever.invoke("What's a highly rated (above 8.5) science fiction film?")

retriever.invoke(
    "What's a movie after 1990 but before 2005 that's all about toys, and preferably is animated"
)

"""
### Filter k

We can also use the self query retriever to specify `k`: the number of documents to fetch.

We can do this by passing `enable_limit=True` to the constructor.
"""
logger.info("### Filter k")

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
)

retriever.invoke("What are two movies about dinosaurs")

"""
## Constructing from scratch with LCEL

To see what's going on under the hood, and to have more custom control, we can reconstruct our retriever from scratch.

First, we need to create a query-construction chain. This chain will take a user query and generated a `StructuredQuery` object which captures the filters specified by the user. We provide some helper functions for creating a prompt and output parser. These have a number of tunable params that we'll ignore here for simplicity.
"""
logger.info("## Constructing from scratch with LCEL")


prompt = get_query_constructor_prompt(
    document_content_description,
    metadata_field_info,
)
output_parser = StructuredQueryOutputParser.from_components()
query_constructor = prompt | llm | output_parser

"""
Let's look at our prompt:
"""
logger.info("Let's look at our prompt:")

logger.debug(prompt.format(query="dummy question"))

"""
And what our full chain produces:
"""
logger.info("And what our full chain produces:")

query_constructor.invoke(
    {
        "query": "What are some sci-fi movies from the 90's directed by Luc Besson about taxi drivers"
    }
)

"""
The query constructor is the key element of the self-query retriever. To make a great retrieval system you'll need to make sure your query constructor works well. Often this requires adjusting the prompt, the examples in the prompt, the attribute descriptions, etc. For an example that walks through refining a query constructor on some hotel inventory data, [check out this cookbook](https://github.com/langchain-ai/langchain/blob/master/cookbook/self_query_hotel_search.ipynb).

The next key element is the structured query translator. This is the object responsible for translating the generic `StructuredQuery` object into a metadata filter in the syntax of the vector store you're using. LangChain comes with a number of built-in translators. To see them all head to the [Integrations section](/docs/integrations/retrievers/self_query).
"""
logger.info(
    "The query constructor is the key element of the self-query retriever. To make a great retrieval system you'll need to make sure your query constructor works well. Often this requires adjusting the prompt, the examples in the prompt, the attribute descriptions, etc. For an example that walks through refining a query constructor on some hotel inventory data, [check out this cookbook](https://github.com/langchain-ai/langchain/blob/master/cookbook/self_query_hotel_search.ipynb).")


retriever = SelfQueryRetriever(
    query_constructor=query_constructor,
    vectorstore=vectorstore,
    structured_query_translator=ChromaTranslator(),
)

retriever.invoke(
    "What's a movie after 1990 but before 2005 that's all about toys, and preferably is animated"
)

logger.info("\n\n[DONE]", bright=True)
