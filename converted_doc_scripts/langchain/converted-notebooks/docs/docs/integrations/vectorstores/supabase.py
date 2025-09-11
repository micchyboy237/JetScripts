from dotenv import load_dotenv
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_text_splitters import CharacterTextSplitter
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

>[Supabase](https://supabase.com/docs) is an open-source Firebase alternative. `Supabase` is built on top of `PostgreSQL`, which offers strong SQL querying capabilities and enables a simple interface with already-existing tools and frameworks.

>[PostgreSQL](https://en.wikipedia.org/wiki/PostgreSQL) also known as `Postgres`, is a free and open-source relational database management system (RDBMS) emphasizing extensibility and SQL compliance.

This notebook shows how to use `Supabase` and `pgvector` as your VectorStore.

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

To run this notebook, please ensure:
- the `pgvector` extension is enabled
- you have installed the `supabase-py` package
- that you have created a `match_documents` function in your database
- that you have a `documents` table in your `public` schema similar to the one below.

The following function determines cosine similarity, but you can adjust to your needs.

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
"""
logger.info("# Supabase (Postgres)")

# %pip install --upgrade --quiet  supabase

"""
We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.
"""
logger.info(
    "We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")

if "SUPABASE_URL" not in os.environ:
    #     os.environ["SUPABASE_URL"] = getpass.getpass("Supabase URL:")

if "SUPABASE_SERVICE_KEY" not in os.environ:
    #     os.environ["SUPABASE_SERVICE_KEY"] = getpass.getpass("Supabase Service Key:")


load_dotenv()

"""
First we'll create a Supabase client and instantiate a Ollama embeddings class.
"""
logger.info(
    "First we'll create a Supabase client and instantiate a Ollama embeddings class.")


supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

"""
Next we'll load and parse some data for our vector store (skip if you already have documents with embeddings stored in your DB).
"""
logger.info("Next we'll load and parse some data for our vector store (skip if you already have documents with embeddings stored in your DB).")


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

"""
Insert the above documents into the database. Embeddings will automatically be generated for each document. You can adjust the chunk_size based on the amount of documents you have. The default is 500 but lowering it may be necessary.
"""
logger.info("Insert the above documents into the database. Embeddings will automatically be generated for each document. You can adjust the chunk_size based on the amount of documents you have. The default is 500 but lowering it may be necessary.")

vector_store = SupabaseVectorStore.from_documents(
    docs,
    embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
    chunk_size=500,
)

"""
Alternatively if you already have documents with embeddings in your database, simply instantiate a new `SupabaseVectorStore` directly:
"""
logger.info("Alternatively if you already have documents with embeddings in your database, simply instantiate a new `SupabaseVectorStore` directly:")

vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

"""
Finally, test it out by performing a similarity search:
"""
logger.info("Finally, test it out by performing a similarity search:")

query = "What did the president say about Ketanji Brown Jackson"
matched_docs = vector_store.similarity_search(query)

logger.debug(matched_docs[0].page_content)

"""
## Similarity search with score

The returned distance score is cosine distance. Therefore, a lower score is better.
"""
logger.info("## Similarity search with score")

matched_docs = vector_store.similarity_search_with_relevance_scores(query)

matched_docs[0]

"""
## Retriever options

This section goes over different options for how to use SupabaseVectorStore as a retriever.

### Maximal Marginal Relevance Searches

In addition to using similarity search in the retriever object, you can also use `mmr`.
"""
logger.info("## Retriever options")

retriever = vector_store.as_retriever(search_type="mmr")

matched_docs = retriever.invoke(query)

for i, d in enumerate(matched_docs):
    logger.debug(f"\n## Document {i}\n")
    logger.debug(d.page_content)

logger.info("\n\n[DONE]", bright=True)
