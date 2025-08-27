import asyncio
from jet.transformers.formatters import format_json
from dataclasses import dataclass
from datetime import datetime
from jet.logger import CustomLogger
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureChatPromptExecutionSettings,
    AzureTextEmbedding,
)
from semantic_kernel.connectors.ai.ollama import (
    OllamaChatCompletion,
    OllamaChatPromptExecutionSettings,
    OllamaTextEmbedding,
)
from semantic_kernel.connectors.postgres import PostgresCollection
from semantic_kernel.contents import ChatHistory
from semantic_kernel.data.vector import (
    DistanceFunction,
    IndexKind,
    VectorStoreField,
    vectorstoremodel,
)
from semantic_kernel.functions import KernelParameterMetadata
from semantic_kernel.functions.kernel_arguments import KernelArguments
from typing import Annotated, Any
import os
import requests
import shutil
import textwrap
import xml.etree.ElementTree as ET


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Using Postgres as memory

This notebook shows how to use Postgres as a memory store in Semantic Kernel.

The code below pulls the most recent papers from [ArviX](https://arxiv.org/), creates embeddings from the paper abstracts, and stores them in a Postgres database.

In the future, we can use the Postgres vector store to search the database for similar papers based on the embeddings - stay tuned!
"""
logger.info("# Using Postgres as memory")


"""
## Set up your environment

You'll need to set up your environment to provide connection information to Postgres, as well as Ollama or Azure Ollama.

To do this, copy the `.env.example` file to `.env` and fill in the necessary information.

__Note__: If you're using VSCode to execute the notebook, the settings in `.env` in the root of the repository will be picked up automatically.

### Postgres configuration

You'll need to provide a connection string to a Postgres database. You can use a local Postgres instance, or a cloud-hosted one.
You can provide a connection string, or provide environment variables with the connection information. See the .env.example file for `POSTGRES_` settings.

#### Using Docker

You can also use docker to bring up a Postgres instance by following the steps below:

Create an `init.sql` that has the following:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Now you can start a postgres instance with the following:

```
docker pull pgvector/pgvector:pg16
docker run --rm -it --name pgvector -p 5432:5432 -v ./init.sql:/docker-entrypoint-initdb.d/init.sql -e POSTGRES_PASSWORD=example pgvector/pgvector:pg16
```

_Note_: Use `.\init.sql` on Windows and `./init.sql` on WSL or Linux/Mac.

Then you could use the connection string:

```
POSTGRES_CONNECTION_STRING="host=localhost port=5432 dbname=postgres user=postgres password=example"
```

### Ollama configuration

You can either use Ollama or Azure Ollama APIs. You provide the API key and other configuration in the `.env` file. Set either the `OPENAI_` or `AZURE_OPENAI_` settings.
"""
logger.info("## Set up your environment")

env_file_path = ".env"

"""
Here we set some additional configuration.
"""
logger.info("Here we set some additional configuration.")

SEARCH_TERM = "RAG"

ARVIX_CATEGORY = "cs.AI"

MAX_RESULTS = 300


USE_AZURE_OPENAI = False

"""
Here we define a vector store model. This model defines the table and column names for storing the embeddings. We use the `@vectorstoremodel` decorator to tell Semantic Kernel to create a vector store definition from the model. The VectorStoreRecordField annotations define the fields that will be stored in the database, including key and vector fields.
"""
logger.info("Here we define a vector store model. This model defines the table and column names for storing the embeddings. We use the `@vectorstoremodel` decorator to tell Semantic Kernel to create a vector store definition from the model. The VectorStoreRecordField annotations define the fields that will be stored in the database, including key and vector fields.")


@vectorstoremodel
@dataclass
class ArxivPaper:
    id: Annotated[str, VectorStoreField("key")]
    title: Annotated[str, VectorStoreField("data")]
    abstract: Annotated[str, VectorStoreField("data")]
    published: Annotated[datetime, VectorStoreField("data")]
    authors: Annotated[list[str], VectorStoreField("data")]
    link: Annotated[str | None, VectorStoreField("data")]
    abstract_vector: Annotated[
        list[float] | str | None,
        VectorStoreField(
            "vector",
            index_kind=IndexKind.HNSW,
            dimensions=1536,
            distance_function=DistanceFunction.COSINE_DISTANCE,
        ),
    ] = None

    def __post_init__(self):
        if self.abstract_vector is None:
            self.abstract_vector = self.abstract

    @classmethod
    def from_arxiv_info(cls, arxiv_info: dict[str, Any]) -> "ArxivPaper":
        return cls(
            id=arxiv_info["id"],
            title=arxiv_info["title"].replace("\n  ", " "),
            abstract=arxiv_info["abstract"].replace("\n  ", " "),
            published=arxiv_info["published"],
            authors=arxiv_info["authors"],
            link=arxiv_info["link"],
        )


"""
Below is a function that queries the ArviX API for the most recent papers based on our search query and category.
"""
logger.info("Below is a function that queries the ArviX API for the most recent papers based on our search query and category.")


def query_arxiv(search_query: str, category: str = "cs.AI", max_results: int = 10) -> list[dict[str, Any]]:
    """
    Query the ArXiv API and return a list of dictionaries with relevant metadata for each paper.

    Args:
        search_query: The search term or topic to query for.
        category: The category to restrict the search to (default is "cs.AI").
        See https://arxiv.org/category_taxonomy for a list of categories.
        max_results: Maximum number of results to retrieve (default is 10).
    """
    response = requests.get(
        "http://export.arxiv.org/api/query?"
        f"search_query=all:%22{search_query.replace(' ', '+')}%22"
        f"+AND+cat:{category}&start=0&max_results={max_results}&sortBy=lastUpdatedDate&sortOrder=descending"
    )

    root = ET.fromstring(response.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    return [
        {
            "id": entry.find("atom:id", ns).text.split("/")[-1],
            "title": entry.find("atom:title", ns).text,
            "abstract": entry.find("atom:summary", ns).text,
            "published": entry.find("atom:published", ns).text,
            "link": entry.find("atom:id", ns).text,
            "authors": [author.find("atom:name", ns).text for author in entry.findall("atom:author", ns)],
            "categories": [category.get("term") for category in entry.findall("atom:category", ns)],
            "pdf_link": next(
                (link_tag.get("href") for link_tag in entry.findall(
                    "atom:link", ns) if link_tag.get("title") == "pdf"),
                None,
            ),
        }
        for entry in root.findall("atom:entry", ns)
    ]


"""
We use this function to query papers and store them in memory as our model types.
"""
logger.info(
    "We use this function to query papers and store them in memory as our model types.")

arxiv_papers: list[ArxivPaper] = [
    ArxivPaper.from_arxiv_info(paper)
    for paper in query_arxiv(SEARCH_TERM, category=ARVIX_CATEGORY, max_results=MAX_RESULTS)
]

logger.debug(f"Found {len(arxiv_papers)} papers on '{SEARCH_TERM}'")

"""
Create a `PostgresCollection`, which represents the table in Postgres where we will store the paper information and embeddings.
"""
logger.info("Create a `PostgresCollection`, which represents the table in Postgres where we will store the paper information and embeddings.")

if USE_AZURE_OPENAI:
    text_embedding = AzureTextEmbedding(
        service_id="embedding", env_file_path=env_file_path)
else:
    text_embedding = OllamaTextEmbedding(
        service_id="embedding", env_file_path=env_file_path)
collection = PostgresCollection[str, ArxivPaper](
    collection_name="arxiv_records",
    record_type=ArxivPaper,
    env_file_path=env_file_path,
    embedding_generator=text_embedding,
)

"""
Now that the models have embeddings, we can write them into the Postgres database.
"""
logger.info(
    "Now that the models have embeddings, we can write them into the Postgres database.")


async def async_func_0():
    async with collection:
        await collection.ensure_collection_exists()

        async def run_async_code_e08d9d16():
            keys = await collection.upsert(arxiv_papers)
            return keys
        keys = asyncio.run(run_async_code_e08d9d16())
        logger.success(format_json(keys))
    return result

result = asyncio.run(async_func_0())
logger.success(format_json(result))

"""
Here we retrieve the first few models from the database and print out their information.
"""
logger.info(
    "Here we retrieve the first few models from the database and print out their information.")


async def async_func_0():
    async with collection:
        async def run_async_code_27c49216():
            results = await collection.get(keys[:3])
            return results
        results = asyncio.run(run_async_code_27c49216())
        logger.success(format_json(results))
        if results:
            for result in results:
                logger.debug(f"# {result.title}")
                logger.debug()
                wrapped_abstract = textwrap.fill(result.abstract, width=80)
                logger.debug(f"Abstract: {wrapped_abstract}")
                logger.debug(f"Published: {result.published}")
                logger.debug(f"Link: {result.link}")
                logger.debug(f"PDF Link: {result.link}")
                logger.debug(f"Authors: {', '.join(result.authors)}")
                logger.debug(f"Embedding: {result.abstract_vector}")
                logger.debug()
                logger.debug()
    return result

result = asyncio.run(async_func_0())
logger.success(format_json(result))

"""
The `VectorStoreTextSearch` object gives us the ability to retrieve semantically similar documents directly from a prompt.
Here we search for the top 5 ArXiV abstracts in our database similar to the query about chunking strategies in RAG applications:
"""
logger.info("The `VectorStoreTextSearch` object gives us the ability to retrieve semantically similar documents directly from a prompt.")

query = "What are good chunking strategies to use for unstructured text in Retrieval-Augmented Generation applications?"


async def async_func_2():
    async with collection:
        async def run_async_code_38fa4ccb():
            search_results = await collection.search(query, top=5, include_total_count=True)
            return search_results
        search_results = asyncio.run(run_async_code_38fa4ccb())
        logger.success(format_json(search_results))
        logger.debug(f"Found {search_results.total_count} results for query.")
        async for search_result in search_results.results:
            title = search_result.record.title
            score = search_result.score
            logger.debug(f"{title}: {score}")
    return result

result = asyncio.run(async_func_2())
logger.success(format_json(result))

"""
We can enable chat completion to utilize the text search by creating a kernel function for searching the database...
"""
logger.info("We can enable chat completion to utilize the text search by creating a kernel function for searching the database...")

kernel = Kernel()
plugin = kernel.add_functions(
    plugin_name="arxiv_plugin",
    functions=[
        collection.create_search_function(
            description="Searches for ArXiv papers that are related to the query.",
            parameters=[
                KernelParameterMetadata(
                    name="query", description="What to search for.", type="str", is_required=True, type_object=str
                ),
                KernelParameterMetadata(
                    name="top",
                    description="Number of results to return.",
                    type="int",
                    default_value=2,
                    type_object=int,
                ),
            ],
        ),
    ],
)

"""
...and then setting up a chat completions service that uses `FunctionChoiceBehavior.Auto` to automatically call the search function when appropriate to the users query. We also create the chat function that will be invoked by the kernel.
"""

chat_completion = OllamaChatCompletion(
    ai_model_id="llama3.2", service_id="completions")
kernel.add_service(chat_completion)

chat_function = kernel.add_function(
    prompt="{{$chat_history}}{{$user_input}}",
    plugin_name="ChatBot",
    function_name="Chat",
)

execution_settings = OllamaChatPromptExecutionSettings(
    function_choice_behavior=FunctionChoiceBehavior.Auto(
        filters={"excluded_plugins": ["ChatBot"]}),
    service_id="chat",
    max_tokens=7000,
    temperature=0.7,
    top_p=0.8,
)

"""
Here we create a chat history with a system message and some initial context:
"""
logger.info(
    "Here we create a chat history with a system message and some initial context:")

history = ChatHistory()
system_message = """
You are a chat bot. Your name is Archie and
you have one goal: help people find answers
to technical questions by relying on the latest
research papers published on ArXiv.
You communicate effectively in the style of a helpful librarian.
You always make sure to include the
ArXiV paper references in your responses.
If you cannot find the answer in the papers,
you will let the user know, but also provide the papers
you did find to be most relevant. If the abstract of the
paper does not specifically reference the user's inquiry,
but you believe it might be relevant, you can still include it
BUT you must make sure to mention that the paper might not directly
address the user's inquiry. Make certain that the papers you link are
from a specific search result.
"""
history.add_system_message(system_message)
history.add_user_message("Hi there, who are you?")
history.add_assistant_message(
    "I am Archie, the ArXiV chat bot. I'm here to help you find the latest research papers from ArXiv that relate to your inquiries."
)

"""
We can now invoke the chat function via the Kernel to get chat completions:
"""
logger.info(
    "We can now invoke the chat function via the Kernel to get chat completions:")

arguments = KernelArguments(
    user_input=query,
    chat_history=history,
    settings=execution_settings,
)


async def run_async_code_d4198ee8():
    result = await kernel.invoke(chat_function, arguments=arguments)
    logger.success(format_json(result))
    return result
result = asyncio.run(run_async_code_d4198ee8())
logger.success(format_json(result))

"""
Printing the result shows that the chat completion service used our text search to locate relevant ArXiV papers based on the query:
"""
logger.info("Printing the result shows that the chat completion service used our text search to locate relevant ArXiV papers based on the query:")


def wrap_text(text, width=90):
    paragraphs = text.split("\n\n")  # Split the text into paragraphs
    wrapped_paragraphs = [
        "\n".join(textwrap.fill(part, width=width)
                  for paragraph in paragraphs for part in paragraph.split("\n"))
    ]  # Wrap each paragraph, split by newlines
    # Join the wrapped paragraphs back together
    return "\n\n".join(wrapped_paragraphs)


logger.debug(f"Archie:>\n{wrap_text(str(result))}")

logger.info("\n\n[DONE]", bright=True)
