from dotenv import load_dotenv
from jet.adapters.langchain.chat_ollama import Ollama
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from supabase.client import Client, create_client
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
# Supabase (Postgres)

>[Supabase](https://supabase.com/docs) is an open-source `Firebase` alternative. 
> `Supabase` is built on top of `PostgreSQL`, which offers strong `SQL` 
> querying capabilities and enables a simple interface with already-existing tools and frameworks.

>[PostgreSQL](https://en.wikipedia.org/wiki/PostgreSQL) also known as `Postgres`,
> is a free and open-source relational database management system (RDBMS) 
> emphasizing extensibility and `SQL` compliance.
>
>[Supabase](https://supabase.com/docs/guides/ai) provides an open-source toolkit for developing AI applications
>using Postgres and pgvector. Use the Supabase client libraries to store, index, and query your vector embeddings at scale.

In the notebook, we'll demo the `SelfQueryRetriever` wrapped around a `Supabase` vector store.

Specifically, we will:
1. Create a Supabase database
2. Enable the `pgvector` extension
3. Create a `documents` table and `match_documents` function that will be used by `SupabaseVectorStore`
4. Load sample documents into the vector store (database table)
5. Build and test a self-querying retriever

## Setup Supabase Database

1. Head over to https://database.new to provision your Supabase database.
2. In the studio, jump to the [SQL editor](https://supabase.com/dashboard/project/_/sql/new) and run the following script to enable `pgvector` and setup your database as a vector store:
    ```sql
    -- Enable the pgvector extension to work with embedding vectors
    create extension if not exists vector;

    -- Create a table to store your documents
    create table
      documents (
        id uuid primary key,
        content text, -- corresponds to Document.pageContent
        metadata jsonb, -- corresponds to Document.metadata
        embedding vector (1536) -- 1536 works for Ollama embeddings, change if needed
      );

    -- Create a function to search for documents
    create function match_documents (
      query_embedding vector (1536),
      filter jsonb default '{}'
    ) returns table (
      id uuid,
      content text,
      metadata jsonb,
      similarity float
    ) language plpgsql as $$
    #variable_conflict use_column
    begin
      return query
      select
        id,
        content,
        metadata,
        1 - (documents.embedding <=> query_embedding) as similarity
      from documents
      where metadata @> filter
      order by documents.embedding <=> query_embedding;
    end;
    $$;
    ```

## Creating a Supabase vector store
Next we'll want to create a Supabase vector store and seed it with some data. We've created a small demo set of documents that contain summaries of movies.

Be sure to install the latest version of `langchain` with `ollama` support:
"""
logger.info("# Supabase (Postgres)")

# %pip install --upgrade --quiet  langchain langchain-ollama tiktoken

"""
The self-query retriever requires you to have `lark` installed:
"""
logger.info("The self-query retriever requires you to have `lark` installed:")

# %pip install --upgrade --quiet  lark

"""
We also need the `supabase` package:
"""
logger.info("We also need the `supabase` package:")

# %pip install --upgrade --quiet  supabase

"""
Since we are using `SupabaseVectorStore` and `OllamaEmbeddings`, we have to load their API keys.

- To find your `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`, head to your Supabase project's [API settings](https://supabase.com/dashboard/project/_/settings/api).
  - `SUPABASE_URL` corresponds to the Project URL
  - `SUPABASE_SERVICE_KEY` corresponds to the `service_role` API key

# - To get your `OPENAI_API_KEY`, navigate to [API keys](https://platform.ollama.com/account/api-keys) on your Ollama account and create a new secret key.
"""
logger.info("Since we are using `SupabaseVectorStore` and `OllamaEmbeddings`, we have to load their API keys.")

# import getpass

if "SUPABASE_URL" not in os.environ:
#     os.environ["SUPABASE_URL"] = getpass.getpass("Supabase URL:")
if "SUPABASE_SERVICE_KEY" not in os.environ:
#     os.environ["SUPABASE_SERVICE_KEY"] = getpass.getpass("Supabase Service Key:")
# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")

"""
_Optional:_ If you're storing your Supabase and Ollama API keys in a `.env` file, you can load them with [`dotenv`](https://github.com/theskumar/python-dotenv).
"""

# %pip install --upgrade --quiet  python-dotenv


load_dotenv()

"""
First we'll create a Supabase client and instantiate a Ollama embeddings class.
"""
logger.info("First we'll create a Supabase client and instantiate a Ollama embeddings class.")



supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

"""
Next let's create our documents.
"""
logger.info("Next let's create our documents.")

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
            "genre": "science fiction",
            "rating": 9.9,
        },
    ),
]

vectorstore = SupabaseVectorStore.from_documents(
    docs,
    embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
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
llm = Ollama(temperature=0)
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

retriever.invoke("Has Greta Gerwig directed any movies about women?")

retriever.invoke("What's a highly rated (above 8.5) science fiction film?")

retriever.invoke(
    "What's a movie after 1990 but before (or on) 2005 that's all about toys, and preferably is animated"
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