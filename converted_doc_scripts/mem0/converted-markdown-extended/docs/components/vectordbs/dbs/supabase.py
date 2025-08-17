import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from mem0 import Memory
import os
import shutil
import { Memory } from "mem0ai/oss"


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
[Supabase](https://supabase.com/) is an open-source Firebase alternative that provides a PostgreSQL database with pgvector extension for vector similarity search. It offers a powerful and scalable solution for storing and querying vector embeddings.

Create a [Supabase](https://supabase.com/dashboard/projects) account and project, then get your connection string from Project Settings > Database. See the [docs](https://supabase.github.io/vecs/hosting/) for details.

### Usage

<CodeGroup>
"""
logger.info("### Usage")


# os.environ["OPENAI_API_KEY"] = "sk-xx"

config = {
    "vector_store": {
        "provider": "supabase",
        "config": {
            "connection_string": "postgresql://user:password@host:port/database",
            "collection_name": "memories",
            "index_method": "hnsw",  # Optional: defaults to "auto"
            "index_measure": "cosine_distance"  # Optional: defaults to "cosine_distance"
        }
    }
}

m = Memory.from_config(config)
messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]
m.add(messages, user_id="alice", metadata={"category": "movies"})

"""

"""


config = {
    vectorStore: {
      provider: "supabase",
      config: {
        collectionName: "memories",
        embeddingModelDims: 1536,
        supabaseUrl: process.env.SUPABASE_URL || "",
        supabaseKey: process.env.SUPABASE_KEY || "",
        tableName: "memories",
      },
    },
}

memory = new Memory(config)

messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about a thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]

async def run_async_code_3ea57109():
    await memory.add(messages, { userId: "alice", metadata: { category: "movies" } })
    return 
 = asyncio.run(run_async_code_3ea57109())
logger.success(format_json())

"""
</CodeGroup>

### SQL Migrations for TypeScript Implementation

The following SQL migrations are required to enable the vector extension and create the memories table:
"""
logger.info("### SQL Migrations for TypeScript Implementation")

-- Enable the vector extension
create extension if not exists vector

-- Create the memories table
create table if not exists memories (
  id text primary key,
  embedding vector(1536),
  metadata jsonb,
  created_at timestamp with time zone default timezone('utc', now()),
  updated_at timestamp with time zone default timezone('utc', now())
)

-- Create the vector similarity search function
create or replace function match_vectors(
  query_embedding vector(1536),
  match_count int,
  filter jsonb default '{}'::jsonb
)
returns table (
  id text,
  similarity float,
  metadata jsonb
)
language plpgsql
as $$
begin
  return query
  select
    t.id::text,
    1 - (t.embedding <=> query_embedding) as similarity,
    t.metadata
  from memories t
  where case
    when filter::text = '{}'::text then true
    else t.metadata @> filter
  end
  order by t.embedding <=> query_embedding
  limit match_count
end
$$

"""
Goto [Supabase](https://supabase.com/dashboard/projects) and run the above SQL migrations inside the SQL Editor.

### Config

Here are the parameters available for configuring Supabase:

<Tabs>
<Tab title="Python">
| Parameter | Description | Default Value |
| --- | --- | --- |
| `connection_string` | PostgreSQL connection string (required) | None |
| `collection_name` | Name for the vector collection | `mem0` |
| `embedding_model_dims` | Dimensions of the embedding model | `1536` |
| `index_method` | Vector index method to use | `auto` |
| `index_measure` | Distance measure for similarity search | `cosine_distance` |
</Tab>
<Tab title="TypeScript">
| Parameter | Description | Default Value |
| --- | --- | --- |
| `collectionName` | Name for the vector collection | `mem0` |
| `embeddingModelDims` | Dimensions of the embedding model | `1536` |
| `supabaseUrl` | Supabase URL | None |
| `supabaseKey` | Supabase key | None |
| `tableName` | Name for the vector table | `memories` |
</Tab>
</Tabs>

### Index Methods

The following index methods are supported:

- `auto`: Automatically selects the best available index method
- `hnsw`: Hierarchical Navigable Small World graph index (faster search, more memory usage)
- `ivfflat`: Inverted File Flat index (good balance of speed and memory)

### Distance Measures

Available distance measures for similarity search:

- `cosine_distance`: Cosine similarity (recommended for most embedding models)
- `l2_distance`: Euclidean distance
- `l1_distance`: Manhattan distance
- `max_inner_product`: Maximum inner product similarity

### Best Practices

1. **Index Method Selection**:
   - Use `hnsw` for fastest search performance when memory is not a constraint
   - Use `ivfflat` for a good balance of search speed and memory usage
   - Use `auto` if unsure, it will select the best method based on your data

2. **Distance Measure Selection**:
   - Use `cosine_distance` for most embedding models (MLX, Hugging Face, etc.)
   - Use `max_inner_product` if your vectors are normalized
   - Use `l2_distance` or `l1_distance` if working with raw feature vectors

3. **Connection String**:
   - Always use environment variables for sensitive information in the connection string
   - Format: `postgresql://user:password@host:port/database`
"""
logger.info("### Config")

logger.info("\n\n[DONE]", bright=True)