from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from jet.logger import CustomLogger
from pgvector.psycopg import register_vector
from psycopg import sql
from pymongo.errors import CollectionInvalid
from pymongo.operations import SearchIndexModel
from statistics import mean, stdev
from typing import List, Tuple
import cohere
import concurrent.futures
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pprint
import psycopg
import pymongo
import random
import shutil
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# AI Database Performance Comparison For AI Workloads: PostgreSQL/PgVector vs MongoDB Atlas Vector Search

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/performance_guidance/vector_database_performance_guidance_mongondb_pgvector.ipynb)
-----


```
Note: This resource is intended to provide performance guidance for AI workloads using vector data within databases, this resoruce is not meant to be an official or comprehensive benchmark, but a guide to help you understand the performance characteristics of the databases within specific search patterns and workloads, enabling you to make an informed decision on which database to use for your AI workload.

Because a database can has been traditionally used for a specific workload, doesn't mean that the database is the best fit for the workload.
```

What this notebook doesn't provide:
- A comprehensive benchmark for all databases
- A cost analysis for the databases and workloads


## **Introduction:**

Welcome to this comprehensive notebook, where we provide performance insights for MongoDB and PostgreSQL—two of the most widely used databases in AI workloads. 

In this session, we analyse the performance results of a variety of search mechanisms, including:

- Vector Search
- Hybrid Search

**What You’ll Learn:**

- PostgreSQL with pgvector:
  - How to set up a PostgreSQL database with the pgvector extension.
  - How to run text, vector, and hybrid searches on PostgreSQL.
- MongoDB Atlas Vector Search:
  - How to set up a MongoDB Atlas database with native Vector Search capabilities.
  - How to execute text, vector, and hybrid searches on MongoDB Atlas.
- AI Workload Overview:
  - This notebook showcases a standard AI workload involving vector embeddings and the retrieval of semantically similar documents. 
  - The system leverages two different vector search solutions:
    - PostgreSQL with pgvector: A powerful extension that integrates vector search capabilities directly into PostgreSQL.
    - MongoDB Atlas Vector Search: A native vector search feature built into MongoDB, optimized for modern, document-based applications.
- AI Workload Metrics:
    - Latency: The time it takes to retrieve the top n results
    - Throughput: The number of queries processed per second
    - P95 Latency: The 95th percentile latency of the queries

**Database Platforms:**

For this performance guidance, we utilize:

- MongoDB Atlas: A fully managed, cloud-native database designed for modern applications.
- Neon: A serverless, fully managed PostgreSQL database optimized for cloud deployments.

Whether your focus is on MongoDB or PostgreSQL, this notebook is designed to help you understand their performance characteristics and guide you in achieving optimal performance for your AI

### Key Information

1. **System Configuration**

### MongoDB Atlas (M30 → M40) vs. Neon (4 → 8 vCPUs) Comparison

#### Important Note on Resource Allocation Disparities

When interpreting the performance results in this notebook, it's essential to consider the significant resource allocation differences between the tested systems:

##### MongoDB Atlas (M30 → M40)
- **Minimum**: 2 vCPUs, 8 GB RAM (M30)
- **Maximum**: 4 vCPUs, 16 GB RAM (M40)

##### Neon PostgreSQL
- **Minimum**: 4 vCPUs, 16 GB RAM
- **Maximum**: 8 vCPUs, 32 GB RAM

This means Neon PostgreSQL has **twice the compute resources** at both minimum and maximum configurations compared to MongoDB Atlas. This resource disparity significantly impacts performance results interpretation in several ways:

1. **Performance per Resource Unit**: If MongoDB shows comparable or better performance despite having fewer resources, this suggests higher efficiency per compute unit.

2. **Cost Considerations**: Higher resource allocation typically incurs higher costs.

3. **Scaling Behavior**: Both systems can scale, but across different resource ranges. Performance gains from scaling might manifest differently due to these distinct scaling ranges.

| **Attribute**                      | **MongoDB Atlas (M30 → M40)**                                                                                                          | **Neon** (Autoscaling: 4 → 8 vCPUs)                                                                                                                                |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **vCPUs**                          | - **Min**: M30 → 2 vCPUs (8 GB RAM) <br/> - **Max**: M40 → 4 vCPUs (16 GB RAM)                                                                 | - **Min**: 4 vCPUs (16 GB RAM) <br/> - **Max**: 8 vCPUs (32 GB RAM)                                                                                                                                         |
| **Memory (RAM)**                  | - **M30**: 8 GB  <br/> - **M40**: 16 GB                                                                                                  | - **Min**: 16 GB <br/> - **Max**: 32 GB                                                                                                                                                                    |
| **Storage**                        | - **M30**: ~40 GB included <br/> - **M40**: ~80 GB included <br/> (Can scale further for additional cost)                                | - Remote “pageserver” stores primary data <br/> - Local disk for temp files: 20 GB or 15 GB × 8 CUs (whichever is higher)                                                                                  |
| **Autoscaling (Compute)**         | - **Cluster Tier Scaling**: can move between M30 and M40 <br/> - **Storage Scaling**: automatically grows storage                        | - **Compute Autoscaling**: 4 to 8 vCPUs <br/> - **Scale to Zero**: optional (after 5 min idle)                                                                                                              |
| **IOPS**                           | ~2000+ on M30, higher on M40                                                                                                             | “Burstable” IOPS from cloud storage <br/> Local File Cache for frequently accessed data                                                                                                                    |
| **Max Connections**               | - ~6000 (M30) <br/> - ~12000 (M40)                                                                                                        | - ~1802 (4 vCPUs) <br/> - ~3604 (8 vCPUs)                                                                                                                                                                  |
| **Scale to Zero**                 | Not supported                                                                                                                            | Optional. If enabled, compute suspends when idle (adds startup latency)                                                                                                                                     |
| **Restart Behavior on Resizing**  | - Moving from M30 to M40 triggers a brief re-provisioning <br/> - Minimal downtime but connections can be interrupted                   | - Autoscaling within 4–8 vCPUs **does not** restart connections <br/> - Editing min/max or toggling Scale to Zero triggers a restart                                                                        |
| **Local Disk for Temp Files**     | Adequate for normal ops; M40 has more local disk                                                                                         | - At least 20 GB local disk, or 15 GB × 8 CUs = 120 GB if that’s higher                                                                                                                                     |
| **Release Updates**               | - Minor updates auto-applied <br/> - Major version upgrades can be scheduled                                                             | - Weekly or scheduled updates <br/> - Manual restart may be needed if Scale to Zero is disabled and you want the latest compute engine updates                                                             |

---

### Key Points

- **Resource Range**  
  - MongoDB Atlas scales from **2 vCPUs/8 GB (M30)** to **4 vCPUs/16 GB (M40)**.  
  - Neon ranges from **4 vCPUs/16 GB** to **8 vCPUs/32 GB**.

- **Closer Parity at M40**  
  - When Atlas scales to M40, it matches Neon’s minimum (4 vCPUs/16 GB), allowing more direct performance comparisons.  
  - Neon can still go beyond M40, up to 8 vCPUs/32 GB, if workload spikes exceed M40 capacity.

- **IOPS and Connections**  
  - Atlas M30→M40 has higher IOPS and connection limits at each tier.  
  - Neon’s IOPS is cloud-based and “burstable,” while connections scale with the CPU (CUs).

In summary, **MongoDB Atlas (M30 → M40)** is closer to **Neon (4 → 8 vCPUs)** than previous tiers, especially at the high end (4 vCPUs/16 GB). 
However, Neon still offers more headroom if your workload demands exceed M40’s capacity.





2. **Data Processing**
   - Uses Wikipedia dataset (100,000 entries) with embeddings(Precision: float32, Dimensions: 768) generated by Cohere
   - JSON data is generated from the dataset and stored in the databases
   - Stores data in both PostgreSQL and MongoDB

3. **Performance Testing**
   - Tests different sizes of concurrent queries (1-400 queries)
   - Tests different insertion batch sizes and speed of insertion

| Operation  | Metric | Description |
|------------|--------|-------------|
| Insertion  | Latency | Time taken to insert the data (average response time) |
|            | Throughput | Number of queries processed per second |
| Retrieval  | Latency | Time taken to retrieve the top n results (average response time) |
|            | Throughput | Number of queries processed per second |
|            | P95 Latency | Time taken to retrieve the top n results for 95% of the queries |

4. **Results Visualization**
   - Interactive animations showing request-response cycles
   - Comparative charts for latency and throughput
   - Performance analysis across different batch sizes

## Part 1: Data Setup

Setting up the performance results dictionary `performance_guidance_results` and the batch sizes to test `CONCURRENT_QUERIES` and `TOTAL_QUERIES`

- `performance_guidance_results` is a dictionary that will store the results of the tests
- `CONCURRENT_QUERIES` is a list of the number of queries that are run concurrently
- `TOTAL_QUERIES` is the total number of queries that are run

Performance Guidance Configuration Example:
When testing with a concurrency level of 10:
- We run 100 iterations
- Each iteration runs 10 concurrent queries
- Total queries = 1,000 queries (TOTAL_ITERATIONS * CONCURRENT_QUERIES)

NOTE: For each concurrency level in CONCURRENT_QUERIES:
1. Run TOTAL_QUERIES iterations
2. In each iteration, execute that many concurrent queries
3. Measure and collect latencies for all queries
"""
logger.info("# AI Database Performance Comparison For AI Workloads: PostgreSQL/PgVector vs MongoDB Atlas Vector Search")

performance_guidance_results = {"PostgreSQL": {}, "MongoDB": {}}

CONCURRENT_QUERIES = [
    1,
    2,
    4,
    5,
    8,
]  # 24, 32, 40, 48, 50, 56, 64, 72, 80, 88, 96, 100, 200, 400

TOTAL_QUERIES = 100

# import getpass


def set_env_securely(var_name, prompt):
#     value = getpass.getpass(prompt)
    os.environ[var_name] = value

"""
### Step 1: Install Libraries

All the libraries are installed using pip and facilitate the sourcing of data, embedding generation, and data visualization.

- `datasets`: Hugging Face library for managing and preprocessing datasets across text, image, and audio (https://huggingface.co/datasets)
- `sentence_transformers`: For creating sentence embeddings for tasks like semantic search and clustering. (https://www.sbert.net/)
- `pandas`: A library for data manipulation and analysis with DataFrames and Series (https://pandas.pydata.org/)
- `matplotlib`: A library for creating static, interactive, and animated data visualizations (https://matplotlib.org/)
- `seaborn`: A library for creating statistical data visualizations (https://seaborn.pydata.org/)
- `cohere`: A library for generating embeddings and accessing the Cohere API or models (https://cohere.ai/)
"""
logger.info("### Step 1: Install Libraries")

# %pip install --upgrade --quiet datasets sentence_transformers pandas matplotlib seaborn cohere

"""
### Step 2: Data Loading

The dataset for the notebook is sourced from the Hugging Face Cohere Wikipedia dataset.

The [Cohere/wikipedia-22-12-en-embeddings](https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings) dataset on Hugging Face comprises English Wikipedia articles embedded using Cohere's multilingual-22-12 model. Each entry includes the article's title, text, URL, Wikipedia ID, view count, paragraph ID, language codes, and a 768-dimensional embedding vector. This dataset is valuable for tasks like semantic search, information retrieval, and NLP model training.

For this notebook, we are using 100,000 rows of the dataset and have removed the id, wiki_id, paragraph_id, langs and views columns.
"""
logger.info("### Step 2: Data Loading")


MAX_ROWS = 100000

dataset = load_dataset(
    "Cohere/wikipedia-22-12-en-embeddings", split="train", streaming=True
)
dataset_segment = dataset.take(MAX_ROWS)

dataset_df = pd.DataFrame(dataset_segment)

dataset_df["json_data"] = dataset_df.apply(
    lambda row: {"title": row["title"], "text": row["text"], "url": row["url"]}, axis=1
)

dataset_df = dataset_df.drop(
    columns=["id", "wiki_id", "paragraph_id", "langs", "views"]
)

dataset_df = dataset_df.rename(columns={"emb": "embedding"})

dataset_df.head(5)

"""
### Step 3: Embedding Generation

We use the Cohere API to generate embeddings for the test queries.

To get the Cohere API key, you can sign up for a free account on the [Cohere website](https://dashboard.cohere.com/welcome/login).
"""
logger.info("### Step 3: Embedding Generation")

set_env_securely("COHERE_API_KEY", "Enter your Cohere API key: ")

"""
Using the Cohere API to generate embeddings for the test queries.

Using the `embed-multilingual-v2.0` model. This is the same model used in the Cohere Wikipedia dataset.

Embedding size is 768 dimensions and the precision is float32.
"""
logger.info("Using the Cohere API to generate embeddings for the test queries.")



co = cohere.Client()


def get_cohere_embeddings(
    sentences: List[str],
    model: str = "embed-multilingual-v2.0",
    input_type: str = "search_document",
) -> Tuple[List[float], List[int]]:
    """
    Generates embeddings for the provided sentences using Cohere's embedding model.

    Args:
    sentences (list of str): List of sentences to generate embeddings for.

    Returns:
    Tuple[List[float], List[int]]: A tuple containing two lists of embeddings (float and int8).
    """
    generated_embedding = co.embed(
        texts=sentences,
        model="embed-multilingual-v2.0",
        input_type="search_document",
        embedding_types=["float"],
    ).embeddings

    return generated_embedding.float[0]

"""
Generate embeddings for the query templates

Store the embeddings in a dictionary for easy access

Note: Doing this to avoid the overhead of generating embeddings for each query in the dataset during the performance analysis process, as this is a time consuming process and expensive in terms of API usage.

Note: Feel free to add more queries to the query_templates list to test the performance of the vector database with a larger number of queries
"""
logger.info("Generate embeddings for the query templates")

query_templates = [
    "When was YouTube officially launched, and by whom?",
    "What is YouTube's slogan introduced after Google's acquisition?",
    "How many hours of videos are collectively watched on YouTube daily?",
    "Which was the first video uploaded to YouTube, and when was it uploaded?",
    "What was the acquisition cost of YouTube by Google, and when was the deal finalized?",
    "What was the first YouTube video to reach one million views, and when did it happen?",
    "What are the three separate branches of the United States government?",
    "Which country has the highest documented incarceration rate and prison population?",
    "How many executions have occurred in the United States since 1977, and which countries have more?",
    "What percentage of the global military spending did the United States account for in 2019?",
    "How is the U.S. president elected?",
    "What cooling system innovation was included in the proposed venues for the World Cup in Qatar?",
    "What lawsuit was filed against Google in June 2020, and what was it about?",
    "How much was Google fined by CNIL in January 2022, and for what reason?",
    "When did YouTube join the NSA's PRISM program, according to reports?",
]

query_embeddings = [
    get_cohere_embeddings(sentences=[query], input_type="search_query")
    for query in query_templates
]

query_embeddings_dict = {
    query: embedding for query, embedding in zip(query_templates, query_embeddings)
}

pd.DataFrame(query_embeddings_dict)

"""
## Part 2: Retrieval Mechanisms with PostgreSQL and PgVector

In this section, we create a PostgreSQL database with the PgVector extension and insert the dataset into the database.

We are also going to implement various search mechanisms on the database to test the performance of the database under certain search patterns and workloads.
Specifically, we are going to implement a semantic search mechanism on the database via vector search and a hybrid search mechanism on the database via vector search and text search.

The table `wikipedia_data` is created with the following columns:
- `id`: The unique identifier for each row
- `title`: The title of the Wikipedia article
- `text`: The text of the Wikipedia article
- `url`: The URL of the Wikipedia article
- `json_data`: The JSON data of the Wikipedia article
- `embedding`: The embedding vector for the Wikipedia article

The table is created with a HNSW index with m=16, ef_construction=64 and cosine similarity (these are the default parameters for the HNSW index in pgvector).
- `HNSW`: Hierarchical Navigable Small World graphs are a type of graph-based index that are used for efficient similarity search.
- `m=16`: The number of edges per node in the graph
- `ef_construction=64`: Short for exploration factor construction, is the number of edges to build during the index construction phase
- `ef_search=100`: Short for exploration factor search, is the number of edges to search during the index search phase
- `cosine similarity`: The similarity metric used for the index (formula: dot product(A, B) / (|A||B|))
- `cosine distance`: The distance metric calculated using cosine similarity (1 - cosine similarity)

We perform a semantic search on the database using a single data point of the query templates and their corresponding embeddings.

### Step 1: Install Libraries

- `pgvector` (0.3.6): A PostgreSQL extension for vector similarity search (https://github.com/pgvector/pgvector)
- `psycopg` (3.2.3): A PostgreSQL database adapter for Python (https://www.psycopg.org/)
"""
logger.info("## Part 2: Retrieval Mechanisms with PostgreSQL and PgVector")

# %pip install --upgrade --quiet pgvector "psycopg[binary]"

"""
### Step 2: Create Postgres Table

- `id`: The unique identifier for each row
- `title`: The title of the Wikipedia article
- `text`: The text of the Wikipedia article
- `url`: The URL of the Wikipedia article
- `json_data`: The JSON data of the Wikipedia article
- `embedding`: The embedding vector for the Wikipedia article


**Key aspect of PostgreSQL table creation:**

- `id`: The unique identifier for each row stored with the data type `bigserial` which is a 64-bit integer and auto-incremented.
- `title`: The title of the Wikipedia article stored with the data type `text` which is a variable character string.
- `text`: The text of the Wikipedia article stored with the data type `text` which is a variable character string.
- `url`: The URL of the Wikipedia article stored with the data type `text` which is a variable character string.
- `json_data`: The JSON data of the Wikipedia article stored with the data type `jsonb` which is a binary formatted JSON data type.
- `embedding`: The embedding vector for the Wikipedia article stored with the data type `vector(768)` which is a provided by pgvector and is of 768 dimensions.

**Optimizing the table for search:**

- `search_vector`: The search vector for the Wikipedia article stored with the data type `tsvector` which is a text search data type in PostgreSQL.
- The expression inside the `GENERATED ALWAYS AS` clause is the text(title and text) to be tokenized and indexed for full-text search.
- Using `coalesce` to handle any null values in the title or text columns.
- `STORED`: This keyword indicates that the `search_vector` column is stored in the table, this avoids the overhead of recalculating the `search_vector` column during queries, and improves performance.


**Extra:**
- The `search_vector` column is computed automatically using the text in the `title` and `text` fields, making full-text search more efficient by avoiding on-the-fly computation.
- The `HNSW` index on the `embedding` column is optimized for ANN queries using cosine similarity, which is crucial for semantic search.
- The `GIN` indexes on both the `json_data` and `search_vector` columns ensure fast query performance on JSONB queries and full-text search, respectively.
"""
logger.info("### Step 2: Create Postgres Table")

def create_table(connection):
    with connection.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS wikipedia_data")

        cur.execute("""
            CREATE TABLE wikipedia_data (
                id bigserial PRIMARY KEY,
                title text,
                text text,
                url text,
                json_data jsonb,
                embedding vector(768),
                search_vector tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', coalesce(title, '') || ' ' || coalesce(text, ''))
                ) STORED
            )
        """)

        cur.execute("""
            CREATE INDEX wikipedia_data_embedding_hnsw_idx
            ON wikipedia_data
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)

        cur.execute("""
            CREATE INDEX wikipedia_data_json_data_gin_idx
            ON wikipedia_data
            USING GIN (json_data);
        """)

        cur.execute("""
            CREATE INDEX wikipedia_data_search_vector_idx
            ON wikipedia_data
            USING GIN (search_vector);
        """)

        logger.debug("Table and indexes created successfully")
        connection.commit()

"""
### Step 4: Define insert function

For inserting JSON data, we convert the Python Dictionary in the `json_data` attribute to a JSON string using the `json.dumps()` function.

This is a serilization process that converts the Python Dictionary in the `json_data` attribute to a JSON string that is stored as binary data in the database.
"""
logger.info("### Step 4: Define insert function")




def insert_data_to_postgres(dataframe, connection, database_type="PostgreSQL"):
    """
    Insert data into the PostgreSQL database.

    Args:
    dataframe (pandas.DataFrame): The dataframe containing the data to insert.
    connection (psycopg.extensions.connection): The connection to the PostgreSQL database.
    database_type (str): The type of database (default: "PostgreSQL").
    """
    start_time = time.time()
    total_rows = len(dataframe)

    try:
        with connection.cursor() as cur:
            data_tuples = []
            for _, row in dataframe.iterrows():
                data_tuple = (
                    row["title"],
                    row["text"],
                    row["url"],
                    json.dumps(row["json_data"]),  # Convert dict to JSON string
                    row["embedding"],
                )
                data_tuples.append(data_tuple)

            if not data_tuples:
                raise ValueError("No valid data tuples to insert")

            cur.executemany(
                """
                INSERT INTO wikipedia_data
                (title, text, url, json_data, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                data_tuples,
            )

            connection.commit()

    except Exception as e:
        logger.debug(f"Error during bulk insert: {e}")
        connection.rollback()
        raise e

    end_time = time.time()
    total_time = end_time - start_time
    rows_per_second = len(data_tuples) / total_time


    if database_type not in performance_guidance_results:
        performance_guidance_results[database_type] = {}

    performance_guidance_results[database_type]["insert_time"] = {
        "total_time": total_time,
        "rows_per_second": rows_per_second,
        "total_rows": total_rows,
    }

set_env_securely("PGHOST", "Enter your PGHOST: ")
set_env_securely("PGDATABASE", "Enter your PGDATABASE: ")
set_env_securely("PGUSER", "Enter your PGUSER: ")
set_env_securely("PGPASSWORD", "Enter your PGPASSWORD: ")

neon_db_host = os.environ["PGHOST"]
neon_db_database = os.environ["PGDATABASE"]
neon_db_user = os.environ["PGUSER"]
neon_db_password = os.environ["PGPASSWORD"]

"""
### Step 5: Insert Data into Postgres
"""
logger.info("### Step 5: Insert Data into Postgres")


try:
    conn = psycopg.connect(
        f"dbname={neon_db_database} user={neon_db_user} password={neon_db_password} host={neon_db_host}"
    )

    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    register_vector(conn)

    create_table(conn)

    insert_data_to_postgres(dataset_df, conn)

except Exception as e:
    logger.debug("Failed to execute:", e)
finally:
    conn.close()
    logger.debug("Connection closed")

"""
### Step 6: Define text search function with postgres
"""
logger.info("### Step 6: Define text search function with postgres")

def text_search_with_postgres(query, connection, top_n=5):
    """
    Perform a full-text search on the precomputed 'search_vector' column of the 'wikipedia_data' table.
    """
    with connection.cursor() as cur:
        cur.execute("SELECT plainto_tsquery('english', %s)", (query,))
        ts_query = cur.fetchone()[0]

        cur.execute(
            """
            SELECT title, text, url, json_data,
                   ts_rank_cd(search_vector, %s) AS rank
            FROM wikipedia_data
            WHERE search_vector @@ %s
            ORDER BY rank DESC
            LIMIT %s
            """,
            (ts_query, ts_query, top_n),
        )

        results = cur.fetchall()

        formatted_results = [
            {
                "title": r[0],
                "text": r[1],
                "url": r[2],
                "json_data": r[3],
                "rank": r[4],
            }
            for r in results
        ]

        return formatted_results

try:
    conn = psycopg.connect(
        f"dbname={neon_db_database} user={neon_db_user} password={neon_db_password} host={neon_db_host}"
    )

    text_search_with_postgres_results = text_search_with_postgres(
        "When was YouTube officially launched, and by whom?", conn
    )
    for result in text_search_with_postgres_results:
        logger.debug(f"\nTitle: {result['title']}")
        logger.debug(f"Text: {result['text']}")
        logger.debug(f"URL: {result['url']}")
        logger.debug(f"JSON Data: {result['json_data']}")
        logger.debug(f"Rank: {result['rank']:.4f}")
        logger.debug("-" * 80)

except Exception as e:
    logger.debug("Failed to connect or execute query:", e)
finally:
    conn.close()
    logger.debug("Connection closed")

"""
### Step 7: Define vector search function with postgres

To avoid exhasuting API key usage, we will fetch the query embedding from the `query_embeddings_dict` dictionary.

In the `vector_search_with_postgres` function, we set the HNSW ef parameter to 100 using the `execute_command` function.

This is to set the exploration factor for the HNSW index to 100. And corresponds to the number of nodes/candidates to search during the index search phase.
A node corresponds to a vector in the index.
"""
logger.info("### Step 7: Define vector search function with postgres")

def vector_search_with_postgres(
    query, connection, top_n=5, filter_key=None, filter_value=None
):
    query_embedding = query_embeddings_dict[query]

    with connection.cursor() as cur:
        cur.execute("SET hnsw.ef_search = 100")
        connection.commit()

        sql_query = """
            SELECT title, text, url, json_data,
                   embedding <=> %s::vector AS similarity
            FROM wikipedia_data
        """

        if filter_key and filter_value:
            sql_query += " WHERE json_data->>%s = %s"

        sql_query += """
            ORDER BY similarity ASC
            LIMIT %s
        """

        params = [query_embedding]
        if filter_key and filter_value:
            params.extend([filter_key, filter_value])
        params.append(top_n)

        cur.execute(sql_query, params)

        results = cur.fetchall()

        formatted_results = [
            {
                "title": r[0],
                "text": r[1],
                "url": r[2],
                "json_data": r[3],
                "similarity": r[4],
            }
            for r in results
        ]

        return formatted_results

try:
    conn = psycopg.connect(
        f"dbname={neon_db_database} user={neon_db_user} password={neon_db_password} host={neon_db_host}"
    )

    query_text = "When was YouTube officially launched, and by whom?"
    results = vector_search_with_postgres(
        query_text, conn, top_n=5, filter_key="title", filter_value="YouTube"
    )

    for result in results:
        logger.debug(f"\nTitle: {result['title']}")
        logger.debug(f"Text: {result['text']}")
        logger.debug(f"URL: {result['url']}")
        logger.debug(f"JSON Data: {result['json_data']}")
        logger.debug(f"Similarity Score: {1- result['similarity']:.4f}")
        logger.debug("-" * 80)

except Exception as e:
    logger.debug("Failed to connect or execute query:", e)
finally:
    conn.close()
    logger.debug("Connection closed")

"""
### Step 8: Define hybrid search function with postgres
"""
logger.info("### Step 8: Define hybrid search function with postgres")



def hybrid_search_with_postgres(
    query, connection, top_n=5, filter_key=None, filter_value=None
):
    """
    Perform a hybrid search combining semantic vector similarity and full-text search.

    Args:
        query (str): The search query string.
        connection: A psycopg2 database connection object.
        top_n (int): Number of top results to return (default is 5).
        filter_key (str, optional): JSON key to filter results on.
        filter_value (str, optional): Value of the JSON key to filter results.

    Returns:
        list: A list of dictionaries containing the search results.
    """
    query_embedding = query_embeddings_dict[query]

    with connection.cursor() as cur:
        cur.execute("SET ivfflat.probes = 10")
        connection.commit()

        base_vector_query = sql.SQL("""
            SELECT id, title, text, url, json_data,
                   embedding <=> %s::vector AS vector_similarity
            FROM wikipedia_data
        """)
        base_full_text_query = sql.SQL("""
            SELECT id, title, text, url, json_data,
                   ts_rank_cd(search_vector, plainto_tsquery('english', %s)) AS text_rank
            FROM wikipedia_data
            WHERE search_vector @@ plainto_tsquery('english', %s)
        """)

        vector_params = [query_embedding]
        text_params = [query, query]

        if filter_key and filter_value:
            filter_condition = sql.SQL("json_data->>{} = %s").format(
                sql.Literal(filter_key)
            )
            base_vector_query += sql.SQL(" WHERE ") + filter_condition
            base_full_text_query += sql.SQL(" AND ") + filter_condition
            vector_params.append(filter_value)
            text_params.append(filter_value)

        cur.execute(base_vector_query + sql.SQL(" LIMIT %s"), vector_params + [top_n])
        vector_results = cur.fetchall()

        cur.execute(base_full_text_query + sql.SQL(" LIMIT %s"), text_params + [top_n])
        text_results = cur.fetchall()

        combined_results = {}
        rrf_k = 60  # RRF parameter; adjust as needed

        for rank, row in enumerate(vector_results, start=1):
            doc_id = row[0]
            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "title": row[1],
                    "text": row[2],
                    "url": row[3],
                    "json_data": row[4],
                    "vector_similarity": row[5],
                    "text_rank": 0,
                    "rrf_score": 0,
                }
            combined_results[doc_id]["rrf_score"] += 1 / (rrf_k + rank)

        for rank, row in enumerate(text_results, start=1):
            doc_id = row[0]
            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "title": row[1],
                    "text": row[2],
                    "url": row[3],
                    "json_data": row[4],
                    "vector_similarity": None,
                    "text_rank": row[5],
                    "rrf_score": 0,
                }
            combined_results[doc_id]["rrf_score"] += 1 / (rrf_k + rank)

        sorted_results = sorted(
            combined_results.values(), key=lambda x: x["rrf_score"], reverse=True
        )

        return sorted_results[:top_n]

try:
    conn = psycopg.connect(
        f"dbname={neon_db_database} user={neon_db_user} password={neon_db_password} host={neon_db_host}"
    )

    query_text = "When was YouTube officially launched, and by whom?"
    results = hybrid_search_with_postgres(
        query_text, conn, top_n=5, filter_key="title", filter_value="YouTube"
    )
    for result in results:
        logger.debug(f"\nTitle: {result['title']}")
        logger.debug(f"Text: {result['text']}")
        logger.debug(f"URL: {result['url']}")
        logger.debug(f"JSON Data: {result['json_data']}")
        if result["vector_similarity"] is not None:
            logger.debug(f"Vector Similarity Score: {1 - result['vector_similarity']:.4f}")
        if result["text_rank"] > 0:
            logger.debug(f"Text Rank: {result['text_rank']:.4f}")
        logger.debug("-" * 80)
except Exception as e:
    logger.debug("Failed to connect or execute query:", e)
finally:
    conn.close()
    logger.debug("Connection closed")

"""
## Part 3: Retrieval Mechanisms with MongoDB Atlas

### Step 1: Install Libraries

- `pymongo` (4.10.1): A Python driver for MongoDB (https://pymongo.readthedocs.io/en/stable/)
"""
logger.info("## Part 3: Retrieval Mechanisms with MongoDB Atlas")

# %pip install --quiet --upgrade pymongo

"""
### Step 2: Create MongoDB Atlas Account

TODO: Place inforioant required

### Step 3: Connect to MongoDB and Create Database and Collection
"""
logger.info("### Step 2: Create MongoDB Atlas Account")

set_env_securely("MONGO_URI", "Enter your MONGO URI: ")

"""
In the following code blocks below we do the following:
1. Establish a connection to the MongoDB database
2. Create a database and collection if they do not already exist
3. Delete all data in the collection if it already exists
"""
logger.info("In the following code blocks below we do the following:")



def get_mongo_client(mongo_uri):
    """Establish and validate connection to the MongoDB."""

    client = pymongo.MongoClient(
        mongo_uri, appname="devrel.showcase.postgres_neon_vs_mongodb_atlas.python"
    )

    ping_result = client.admin.command("ping")
    if ping_result.get("ok") == 1.0:
        logger.debug("Connection to MongoDB successful")
        return client
    else:
        logger.debug("Connection to MongoDB failed")
    return None


MONGO_URI = os.environ["MONGO_URI"]
if not MONGO_URI:
    logger.debug("MONGO_URI not set in environment variables")


mongo_client = get_mongo_client(MONGO_URI)

DB_NAME = "vector_db"
COLLECTION_NAME = "wikipedia_data"

db = mongo_client[DB_NAME]

if COLLECTION_NAME not in db.list_collection_names():
    try:
        db.create_collection(COLLECTION_NAME)
        logger.debug(f"Collection '{COLLECTION_NAME}' created successfully.")
    except CollectionInvalid as e:
        logger.debug(f"Error creating collection: {e}")
else:
    logger.debug(f"Collection '{COLLECTION_NAME}' already exists.")

collection = db[COLLECTION_NAME]

collection.delete_many({})

"""
### Step 4: Vector Index Creation

The `setup_vector_search_index` function creates a vector search index for the MongoDB collection.

The `index_name` parameter is the name of the index to create.

The `embedding_field_name` parameter is the name of the field containing the text embeddings on each document within the wikipedia_data collection.
"""
logger.info("### Step 4: Vector Index Creation")

embedding_field_name = "embedding"
vector_search_index_name = "vector_index"

"""
Filtering your data is useful to narrow the scope of your semantic search and ensure that not all vectors are considered for comparison. It reduces the number of documents against which to run similarity comparisons, which can decrease query latency and increase the accuracy of search results.

You must index the fields that you want to filter by using the filter type inside the fields array.
"""
logger.info("Filtering your data is useful to narrow the scope of your semantic search and ensure that not all vectors are considered for comparison. It reduces the number of documents against which to run similarity comparisons, which can decrease query latency and increase the accuracy of search results.")




def setup_vector_search_index(collection, index_name="vector_index"):
    """
    Setup a vector search index for a MongoDB collection.

    Args:
    collection: MongoDB collection object
    index_definition: Dictionary containing the index definition
    index_name: Name of the index (default: "vector_index")
    """
    new_vector_search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 768,
                    "similarity": "cosine",
                },
                {
                    "type": "filter",
                    "path": "json_data.title",
                },
            ]
        },
        name=index_name,
        type="vectorSearch",
    )

    try:
        result = collection.create_search_index(model=new_vector_search_index_model)
        logger.debug(f"Creating index '{index_name}'...")

        return result

    except Exception as e:
        logger.debug(f"Error creating new vector search index '{index_name}': {e!s}")
        return None

setup_vector_search_index(collection, "vector_index")

"""
An Atlas Search index is a data structure that categorizes data in an easily searchable format. It is a mapping between terms and the documents that contain those terms. Atlas Search indexes enable faster retrieval of documents using certain identifiers. You must configure an Atlas Search index to query data in your Atlas cluster using Atlas Search.

You can create an Atlas Search index on a single field or on multiple fields. We recommend that you index the fields that you regularly use to sort or filter your data in order to quickly retrieve the documents that contain the relevant data at query-time.
"""
logger.info("An Atlas Search index is a data structure that categorizes data in an easily searchable format. It is a mapping between terms and the documents that contain those terms. Atlas Search indexes enable faster retrieval of documents using certain identifiers. You must configure an Atlas Search index to query data in your Atlas cluster using Atlas Search.")

def setup_text_search_index(collection, index_name="text_search_index"):
    """
    Setup a text search index for a MongoDB collection in Atlas.

    Args:
        uri (str): MongoDB Atlas connection string.
        database_name (str): Name of the database.
        collection_name (str): Name of the collection.
        index_name (str): Name of the index (default: "text_search_index").
    """
    search_index_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": True  # Index all fields dynamically
            },
        },
        name=index_name,
        type="search",
    )

    try:
        result = collection.create_search_index(model=search_index_model)
        logger.debug(f"Creating index '{index_name}'...")

        time.sleep(30)
        logger.debug(f"30-second wait completed for index '{index_name}'.")

        logger.debug(f"Index '{index_name}' created successfully.")
        return result

    except Exception as e:
        logger.debug(f"Error creating text search index '{index_name}': {e}")
        return None

setup_text_search_index(collection, "text_search_index")

"""
### Step 5: Define Insert Data Function

Because of the affinity of MongoDB for JSON data, we don't have to convert the Python Dictionary in the `json_data` attribute to a JSON string using the `json.dumps()` function. Instead, we can directly insert the Python Dictionary into the MongoDB collection.

This reduced the operational overhead of the insertion processes in AI workloads.
"""
logger.info("### Step 5: Define Insert Data Function")

def insert_data_to_mongodb(dataframe, collection, database_type="MongoDB"):
    start_time = time.time()
    total_rows = len(dataframe)

    try:
        documents = dataframe.to_dict("records")

        result = collection.insert_many(documents)

        end_time = time.time()
        total_time = end_time - start_time
        rows_per_second = total_rows / total_time


        if database_type not in performance_guidance_results:
            performance_guidance_results[database_type] = {}

        performance_guidance_results[database_type]["insert_time"] = {
            "total_time": total_time,
            "rows_per_second": rows_per_second,
            "total_rows": total_rows,
        }

        return True

    except Exception as e:
        logger.debug(f"Error during MongoDB insertion: {e}")
        return False

"""
### Step 6: Insert Data into MongoDB
"""
logger.info("### Step 6: Insert Data into MongoDB")

documents = dataset_df.to_dict("records")
success = insert_data_to_mongodb(dataset_df, collection)

logger.debug(performance_guidance_results["MongoDB"])

"""
### Step 7: Define Text Search Function

The `text_search_with_mongodb` function performs a text search in the MongoDB collection based on the user query.

- `query_text` parameter is the user's query string.
- `collection` parameter is the MongoDB collection to search.
- `top_n` parameter is the number of top results to return.
"""
logger.info("### Step 7: Define Text Search Function")

def text_search_with_mongodb(query_text, collection, top_n=5):
    """
    Perform a text search in the MongoDB collection based on the user query.

    Args:
        query_text (str): The user's query string.
        collection (MongoCollection): The MongoDB collection to search.
        top_n (int): The number of top results to return.

    Returns:
    list: A list of matching documents.
    """
    text_search_stage = {
        "$search": {
            "index": "text_search_index",
            "text": {"query": query_text, "path": "title"},
        }
    }

    limit_stage = {"$limit": top_n}

    project_stage = {
        "$project": {
            "_id": 0,
            "title": 1,
            "text": 1,
            "url": 1,
            "json_data": 1,
        }
    }

    pipeline = [text_search_stage, limit_stage, project_stage]

    results = collection.aggregate(pipeline)
    return list(results)

query_text = "When was YouTube officially launched, and by whom?"

get_knowledge = text_search_with_mongodb(query_text, collection)

pd.DataFrame(get_knowledge).head()

"""
### Step 8: Define Vector Search Function

The `semantic_search_with_mongodb` function performs a vector search in the MongoDB collection based on the user query.

- `user_query` parameter is the user's query string.
- `collection` parameter is the MongoDB collection to search.
- `top_n` parameter is the number of top results to return.
- `vector_search_index_name` parameter is the name of the vector search index to use for the search.

The `numCandidates` parameter is the number of candidate matches to consider. This is set to 100 to match the number of candidate matches to consider in the PostgreSQL vector search.

Another point to note is the queries in MongoDB are performed using the `aggregate` function enabled by the MongoDB Query Language(MQL).

This allows for more flexibility in the queries and the ability to perform more complex searches. And data processing opreations can be defined as stages in the pipeline. If you are a data engineer, data scientist or ML Engineer, the concept of pipeline processing is a key concept.
"""
logger.info("### Step 8: Define Vector Search Function")

def vector_search_with_mongodb(
    user_query,
    collection,
    top_k=5,
    vector_search_index_name="vector_index",
    title_filter=None,
):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.
    additional_stages (list): Additional aggregation stages to include in the pipeline.
    vector_search_index_name (str): The name of the vector search index.

    Returns:
    list: A list of matching documents.
    """

    query_embedding = query_embeddings_dict[user_query]

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    vector_search_stage = {
        "$vectorSearch": {
            "index": vector_search_index_name,  # specifies the index to use for the search
            "queryVector": query_embedding,  # the vector representing the query
            "path": "embedding",  # field in the documents containing the vectors to search against
            "filter": {"json_data.title": title_filter},
            "numCandidates": 100,  # number of candidate matches to consider
            "limit": top_k,  # return top n matches
        }
    }

    project_stage = {
        "$project": {
            "_id": 0,  # Exclude the _id field
            "title": 1,
            "text": 1,
            "url": 1,
            "json_data": 1,
            "score": {
                "$meta": "vectorSearchScore"  # Include the search score
            },
        }
    }

    pipeline = [vector_search_stage, project_stage]

    results = collection.aggregate(pipeline)
    return list(results)

query_text = "When was YouTube officially launched, and by whom?"

get_knowledge = vector_search_with_mongodb(
    query_text, collection, title_filter="YouTube"
)

pd.DataFrame(get_knowledge).head()

"""
### Step 9: Define Hybrid Search Function

The `hybrid_search_with_mongodb` function conducts a hybrid search on a MongoDB Atlas collection that combines a vector search and a full-text search using Atlas Search.

In the MongoDB hybrid search function, there are two weights:

- vector_weight = 0.5: This weight scales the score obtained from the vector search portion.
- full_text_weight = 0.5: This weight scales the score from the full-text search portion.

#### Note: In the MongoDB hybrid search function, two weights:
    - `vector_weight` 
    - `full_text_weight` 

They are used to control the influence of each search component on the final score. 

Here's how they work:

Purpose:
The weights allow you to adjust how much the vector (semantic) search and the full-text search contribute to the overall ranking. 
For example, a higher full_text_weight means that the full-text search results will have a larger impact on the final score, whereas a higher vector_weight would give more importance to the vector similarity score.

Usage in the Pipeline:
Within the aggregation pipeline, after retrieving results from each search type, the function computes a reciprocal ranking score for each result (using an expression like `1/(rank + 60)`). 
This score is then multiplied by the corresponding weight:

**Vector Search:**

```
"vs_score": {
  "$multiply": [ vector_weight, { "$divide": [1.0, { "$add": ["$rank", 60] } ] } ]
}
```


**Full-Text Search:**
```
"fts_score": {
  "$multiply": [ full_text_weight, { "$divide": [1.0, { "$add": ["$rank", 60] } ] } ]
}
```

Finally, these weighted scores are combined (typically by adding them together) to produce a final score that determines the ranking of the documents.

**Impact:**
By adjusting these weights, you can fine-tune the search results to better match your application's needs. For instance, if the full-text component is more reliable for your dataset, you might set full_text_weight higher than vector_weight.

The weights in the MongoDB function allow you to balance the contributions from vector-based and full-text search components, ensuring that the final ranking score reflects the desired importance of each search method.
"""
logger.info("### Step 9: Define Hybrid Search Function")

def hybrid_search_with_mongodb(
    user_query,
    collection,
    vector_search_index_name="vector_index",
    text_search_index_name="text_search_index",
    vector_weight=0.5,
    full_text_weight=0.5,
    top_k=10,
):
    """
    Conduct a hybrid search on a MongoDB Atlas collection that combines a vector search
    and a full-text search using Atlas Search.

    Args:
        user_query (str): The user's query string.
        collection (MongoCollection): The MongoDB collection to search.
        vector_search_index_name (str): The name of the vector search index.
        text_search_index_name (str): The name of the text search index.
        vector_weight (float): The weight of the vector search.
        full_text_weight (float): The weight of the full-text search.

    Returns:
        list: A list of documents (dict) with combined scores.
    """

    collection_name = "wikipedia_data"
    query_vector = query_embeddings_dict[user_query]

    pipeline = [
        {
            "$vectorSearch": {
                "index": vector_search_index_name,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": top_k,
            }
        },
        {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
        {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
        {
            "$addFields": {
                "vs_score": {
                    "$multiply": [
                        vector_weight,
                        {"$divide": [1.0, {"$add": ["$rank", 60]}]},
                    ]
                }
            }
        },
        {"$project": {"vs_score": 1, "_id": "$docs._id", "title": "$docs.title"}},
        {
            "$unionWith": {
                "coll": collection_name,
                "pipeline": [
                    {
                        "$search": {
                            "index": text_search_index_name,
                            "text": {"query": user_query, "path": "title"},
                        }
                    },
                    {"$limit": top_k},
                    {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
                    {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
                    {
                        "$addFields": {
                            "fts_score": {
                                "$multiply": [
                                    full_text_weight,
                                    {"$divide": [1.0, {"$add": ["$rank", 60]}]},
                                ]
                            }
                        }
                    },
                    {
                        "$project": {
                            "fts_score": 1,
                            "_id": "$docs._id",
                            "title": "$docs.title",
                        }
                    },
                ],
            }
        },
        {
            "$group": {
                "_id": "$_id",
                "title": {"$first": "$title"},
                "vs_score": {"$max": "$vs_score"},
                "fts_score": {"$max": "$fts_score"},
            }
        },
        {
            "$project": {
                "_id": 1,
                "title": 1,
                "vs_score": {"$ifNull": ["$vs_score", 0]},
                "fts_score": {"$ifNull": ["$fts_score", 0]},
            }
        },
        {
            "$project": {
                "score": {"$add": ["$fts_score", "$vs_score"]},
                "_id": 1,
                "title": 1,
                "url": 1,
                "text": 1,
                "json_data": 1,
                "vs_score": 1,
                "fts_score": 1,
            }
        },
        {"$sort": {"score": -1}},
        {"$limit": top_k},
    ]

    results = list(collection.aggregate(pipeline))
    return results

query_text = "When was YouTube officially launched, and by whom?"

results = hybrid_search_with_mongodb(
    query_text, collection, vector_weight=0.1, full_text_weight=0.9, top_k=10
)

pd.DataFrame(results).head()

"""
## Part 4: Vector Database Performance Analysis

### 1. Insertion Performance Analysis Process

We are inserting data incrementally with doubling batch sizes and record performance metrics.
Notably, we will be measuring the time it takes to insert data incrementally and the number of rows inserted per second.

We are using the `insert_data_incrementally` function to insert data incrementally.

It starts with a batch size of 1 and doubles the batch size until it has inserted all the data, recording the time it takes to insert the data and the number of rows inserted per second.

The key component we are interested in is the time it takes to insert the data and the number of rows inserted per second. In AI Workloads, there are data ingestion processes that are performned in batches from various data sources. So in practice, we are interested in the time it takes to insert the data and the number of rows inserted per second.
"""
logger.info("## Part 4: Vector Database Performance Analysis")



def insert_data_incrementally(dataframe, connection, database_type="PostgreSQL"):
    """
    Insert data incrementally with doubling batch sizes and record performance metrics.
    """
    incremental_metrics = {}
    total_rows = len(dataframe)
    remaining_rows = total_rows
    start_idx = 0

    batch_sizes = [1, 10]
    current_size = 20
    while current_size < total_rows:
        batch_sizes.append(current_size)
        current_size *= 2

    for batch_size in batch_sizes:
        if remaining_rows <= 0:
            break

        actual_batch_size = min(batch_size, remaining_rows)
        end_idx = start_idx + actual_batch_size

        batch_df = dataframe.iloc[start_idx:end_idx]

        start_time = time.time()

        try:
            if database_type == "PostgreSQL":
                insert_data_to_postgres(batch_df, connection, database_type)
            else:  # MongoDB
                insert_data_to_mongodb(batch_df, connection, database_type)

            end_time = time.time()
            total_time = end_time - start_time
            rows_per_second = actual_batch_size / total_time

            incremental_metrics[actual_batch_size] = {
                "total_time": total_time,
                "rows_per_second": rows_per_second,
                "batch_size": actual_batch_size,
            }


        except Exception as e:
            logger.debug(f"Error during batch insertion (size {batch_size}): {e}")
            raise e

        start_idx = end_idx
        remaining_rows -= actual_batch_size

    if database_type not in performance_guidance_results:
        performance_guidance_results[database_type] = {}

    performance_guidance_results[database_type]["incremental_insert"] = (
        incremental_metrics
    )

    return incremental_metrics

"""
#### 1.1 PostgreSQL Insertion Performance Analysis
"""
logger.info("#### 1.1 PostgreSQL Insertion Performance Analysis")


try:
    conn = psycopg.connect(
        f"dbname={neon_db_database} user={neon_db_user} password={neon_db_password} host={neon_db_host}"
    )
    register_vector(conn)

    create_table(conn)

    postgres_metrics = insert_data_incrementally(dataset_df, conn, "PostgreSQL")

except Exception as e:
    logger.debug("Failed to execute:", e)
finally:
    conn.close()
    logger.debug("\nConnection closed")

"""
#### 1.2 MongoDB Insertion Performance Analysis
"""
logger.info("#### 1.2 MongoDB Insertion Performance Analysis")

try:
    mongo_client = get_mongo_client(MONGO_URI)
    db = mongo_client[DB_NAME]
    collection = db[COLLECTION_NAME]

    collection.delete_many({})

    mongo_metrics = insert_data_incrementally(dataset_df, collection, "MongoDB")

except Exception as e:
    logger.debug("MongoDB operation failed:", e)
finally:
    mongo_client.close()
    logger.debug("\nMongoDB connection closed")

"""
#### 1.3 Visualize Insertion Performance Analysis
"""
logger.info("#### 1.3 Visualize Insertion Performance Analysis")



def plot_combined_insertion_metrics(postgres_metrics, mongo_metrics):
    """
    Creates a combined line plot comparing PostgreSQL and MongoDB insertion metrics.
    """
    plt.figure(figsize=(12, 6))

    batch_sizes = [
        1,
        10,
        20,
        40,
        80,
        160,
        320,
        640,
        1280,
        2560,
        5120,
        10240,
        20480,
        40960,
    ]
    postgres_times = [
        postgres_metrics[size]["total_time"]
        for size in batch_sizes
        if size in postgres_metrics
    ]
    mongo_times = [
        mongo_metrics[size]["total_time"]
        for size in batch_sizes
        if size in mongo_metrics
    ]

    plt.plot(
        batch_sizes[: len(postgres_times)],
        postgres_times,
        marker="o",
        label="PostgreSQL",
        color="blue",
        linewidth=2,
    )
    plt.plot(
        batch_sizes[: len(mongo_times)],
        mongo_times,
        marker="s",
        label="MongoDB",
        color="green",
        linewidth=2,
    )

    plt.title("Database Insertion Time Comparison", fontsize=14)
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=10)

    plt.xscale("log", base=2)

    custom_ticks = batch_sizes
    plt.xticks(custom_ticks, custom_ticks, rotation=45, ha="right")

    for i, (size, time) in enumerate(
        zip(batch_sizes[: len(postgres_times)], postgres_times)
    ):
        plt.annotate(
            f"{time:.1f}s",
            (size, time),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    for i, (size, time) in enumerate(zip(batch_sizes[: len(mongo_times)], mongo_times)):
        plt.annotate(
            f"{time:.1f}s",
            (size, time),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=8,
        )

    postgres_throughput = [
        metrics["rows_per_second"] for metrics in postgres_metrics.values()
    ]
    mongo_throughput = [
        metrics["rows_per_second"] for metrics in mongo_metrics.values()
    ]

    text_info = (
        f"Max Throughput:\n"
        f"PostgreSQL: {max(postgres_throughput):.0f} rows/s\n"
        f"MongoDB: {max(mongo_throughput):.0f} rows/s"
    )

    plt.text(
        0.02,
        0.98,
        text_info,
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
        verticalalignment="top",
        fontsize=10,
    )

    plt.tight_layout()
    plt.show()

plot_combined_insertion_metrics(postgres_metrics, mongo_metrics)

"""
### 2. Semantic Search with PostgreSQL and PgVector Performance Analysis
"""
logger.info("### 2. Semantic Search with PostgreSQL and PgVector Performance Analysis")




def performance_analysis_search_postgres(
    connection,
    search_fn,
    result_key,
    database_type="PostgreSQL",
    num_queries=100,
    concurrent_queries=[1, 10, 50, 100],
):
    """
    Performance Analysis PostgreSQL search performance (vector or hybrid) with concurrent queries.

    Args:
        connection: Database connection.
        search_fn: The search function to analyse.
        result_key: Key for storing results ('vector' or 'hybrid').
        database_type: Type of database (default 'PostgreSQL').
        num_queries: Number of performance analysis iterations.
        concurrent_queries: List of concurrency levels to test.
    """
    query_templates = [
        "When was YouTube officially launched, and by whom?",
        "What is YouTube's slogan introduced after Google's acquisition?",
        "How many hours of videos are collectively watched on YouTube daily?",
        "Which was the first video uploaded to YouTube, and when was it uploaded?",
        "What was the acquisition cost of YouTube by Google, and when was the deal finalized?",
        "What was the first YouTube video to reach one million views, and when did it happen?",
        "What are the three separate branches of the United States government?",
        "Which country has the highest documented incarceration rate and prison population?",
        "How many executions have occurred in the United States since 1977, and which countries have more?",
        "What percentage of the global military spending did the United States account for in 2019?",
        "How is the U.S. president elected?",
        "What cooling system innovation was included in the proposed venues for the World Cup in Qatar?",
        "What lawsuit was filed against Google in June 2020, and what was it about?",
        "How much was Google fined by CNIL in January 2022, and for what reason?",
        "When did YouTube join the NSA's PRISM program, according to reports?",
    ]

    if database_type not in performance_guidance_results:
        performance_guidance_results[database_type] = {}
    performance_guidance_results[database_type][result_key] = {}
    performance_guidance_results[database_type][result_key]["specific"] = {}

    def execute_single_query():
        """Execute a single query and measure its latency."""
        query = random.choice(query_templates)
        start_time = time.time()
        search_fn(query, connection, top_n=5)
        end_time = time.time()
        return end_time - start_time

    for number_of_queries in concurrent_queries:
        latencies = []
        for _ in range(num_queries):
            with ThreadPoolExecutor(max_workers=number_of_queries) as executor:
                futures = [
                    executor.submit(execute_single_query)
                    for _ in range(number_of_queries)
                ]
                batch_latencies = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]
                latencies.extend(batch_latencies)

        avg_latency = mean(latencies)
        throughput = 1 / avg_latency  # Base queries per second per query.
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        std_dev_latency = stdev(latencies)

        performance_guidance_results[database_type][result_key]["specific"][
            number_of_queries
        ] = {
            "avg_latency": avg_latency,
            "throughput": throughput
            * number_of_queries,  # Scale throughput by concurrency.
            "p95_latency": p95_latency,
            "std_dev": std_dev_latency,
        }

    return performance_guidance_results

CONCURRENT_QUERIES = [1, 2, 4, 5, 8]

try:
    conn = psycopg.connect(
        f"dbname={neon_db_database} user={neon_db_user} password={neon_db_password} host={neon_db_host}"
    )
    register_vector(conn)

    logger.debug("Running performance analysis for vector search...")
    performance_analysis_search_postgres(
        conn,
        vector_search_with_postgres,
        "vector",
        database_type="PostgreSQL",
        num_queries=TOTAL_QUERIES,
        concurrent_queries=CONCURRENT_QUERIES,
    )

    logger.debug("Running performance analysis for hybrid search...")
    performance_analysis_search_postgres(
        conn,
        hybrid_search_with_postgres,
        "hybrid",
        database_type="PostgreSQL",
        num_queries=TOTAL_QUERIES,
        concurrent_queries=CONCURRENT_QUERIES,
    )

except Exception as e:
    logger.debug("Performance Analysis failed:", e)
finally:
    conn.close()
    logger.debug("\nConnection closed")


pprint.plogger.debug(performance_guidance_results["PostgreSQL"])

"""
#### 2.2 MongoDB Semantic Search Performance Analysis
"""
logger.info("#### 2.2 MongoDB Semantic Search Performance Analysis")

def performance_analysis_search_mongo(
    collection,
    search_fn,
    result_key,
    database_type="MongoDB",
    num_queries=100,
    concurrent_queries=[1, 10, 50, 100],
):
    """
    MongoDB search performance (vector or hybrid) with true concurrency.

    Args:
        collection: MongoDB collection object.
        search_fn: The search function to analyse. It should accept a query string.
                   For example:
                     - For vector search: lambda q: vector_search_with_mongodb(q, collection, top_n=5)
                     - For hybrid search: lambda q: hybrid_search_with_mongodb(collection, q)
        result_key: Key to store results under (e.g., "vector" or "hybrid").
        database_type: Type of database (default: "MongoDB").
        num_queries: Number of performance analysis iterations for statistical significance.
        concurrent_queries: Different concurrency levels to test.

    Returns:
        The updated performance analysis results for the specified database type.
    """
    query_templates = [
        "When was YouTube officially launched, and by whom?",
        "What is YouTube's slogan introduced after Google's acquisition?",
        "How many hours of videos are collectively watched on YouTube daily?",
        "Which was the first video uploaded to YouTube, and when was it uploaded?",
        "What was the acquisition cost of YouTube by Google, and when was the deal finalized?",
        "What was the first YouTube video to reach one million views, and when did it happen?",
        "What are the three separate branches of the United States government?",
        "Which country has the highest documented incarceration rate and prison population?",
        "How many executions have occurred in the United States since 1977, and which countries have more?",
        "What percentage of the global military spending did the United States account for in 2019?",
        "How is the U.S. president elected?",
        "What cooling system innovation was included in the proposed venues for the World Cup in Qatar?",
        "What lawsuit was filed against Google in June 2020, and what was it about?",
        "How much was Google fined by CNIL in January 2022, and for what reason?",
        "When did YouTube join the NSA's PRISM program, according to reports?",
    ]

    if database_type not in performance_guidance_results:
        performance_guidance_results[database_type] = {}
    performance_guidance_results[database_type][result_key] = {}
    performance_guidance_results[database_type][result_key]["specific"] = {}

    def execute_single_query():
        """Execute a single query using the provided search function and measure its latency."""
        query = random.choice(query_templates)
        start_time = time.time()
        result = search_fn(query, collection)
        end_time = time.time()
        return end_time - start_time

    for number_of_queries in concurrent_queries:
        latencies = []
        for _ in range(num_queries):
            with ThreadPoolExecutor(max_workers=number_of_queries) as executor:
                futures = [
                    executor.submit(execute_single_query)
                    for _ in range(number_of_queries)
                ]
                batch_latencies = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]
                latencies.extend(batch_latencies)

        avg_latency = mean(latencies)
        throughput = 1 / avg_latency  # Base queries per second per query.
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        std_dev_latency = stdev(latencies)

        performance_guidance_results[database_type][result_key]["specific"][
            number_of_queries
        ] = {
            "avg_latency": avg_latency,
            "throughput": throughput
            * number_of_queries,  # Scale throughput by concurrency.
            "p95_latency": p95_latency,
            "std_dev": std_dev_latency,
        }

    return performance_guidance_results[database_type]

try:
    mongo_client = get_mongo_client(MONGO_URI)
    db = mongo_client[DB_NAME]
    collection = db[COLLECTION_NAME]

    logger.debug("Running MongoDB performance analysis for vector search...")
    performance_analysis_search_mongo(
        collection,
        search_fn=vector_search_with_mongodb,
        result_key="vector",
        num_queries=TOTAL_QUERIES,
        concurrent_queries=CONCURRENT_QUERIES,
    )

    logger.debug("Running MongoDB performance analysis for hybrid search...")
    performance_analysis_search_mongo(
        collection,
        search_fn=hybrid_search_with_mongodb,
        result_key="hybrid",
        num_queries=TOTAL_QUERIES,
        concurrent_queries=CONCURRENT_QUERIES,
    )

except Exception as e:
    logger.debug("MongoDB performance analysis failed:", e)
finally:
    mongo_client.close()
    logger.debug("\nMongoDB connection closed")

logger.debug(performance_guidance_results)

"""
#### 2.3 Visualize Vector Search Performance Analysis
"""
logger.info("#### 2.3 Visualize Vector Search Performance Analysis")

def bar_chart_performance_analysis_comparison(
    performance_guidance_results,
    metric="avg_latency",
    metric_label="Average Latency (ms)",
):
    """
    Generates bar charts to compare performance analysis results for each metric across databases,
    for each search type ("vector" and "hybrid") using the 'specific' results.
    """
    search_types = ["vector", "hybrid"]
    batch_sizes = sorted(
        list(
            next(iter(performance_guidance_results.values()))["vector"][
                "specific"
            ].keys()
        )
    )

    for search_type in search_types:
        data = []
        for batch_size in batch_sizes:
            row = {"Batch Size": batch_size}
            for db_type in performance_guidance_results:
                try:
                    value = (
                        performance_guidance_results[db_type]
                        .get(search_type, {})
                        .get("specific", {})
                        .get(batch_size, {})
                        .get(metric, 0)
                    )
                    if value is not None and metric in ["avg_latency", "p95_latency"]:
                        value *= 1000  # Convert seconds to ms
                    row[db_type] = value if value is not None else 0
                except Exception:
                    row[db_type] = 0
            data.append(row)

        labels = [str(row["Batch Size"]) for row in data]
        postgres_values = [row.get("PostgreSQL", 0) for row in data]
        mongodb_values = [row.get("MongoDB", 0) for row in data]

        fig, ax = plt.subplots(figsize=(15, 6))
        width = 0.35
        x = np.arange(len(labels))

        postgres_bars = ax.bar(
            x - width / 2,
            postgres_values,
            width,
            label="PostgreSQL",
            color="lightblue",
            edgecolor="blue",
        )
        mongodb_bars = ax.bar(
            x + width / 2,
            mongodb_values,
            width,
            label="MongoDB",
            color="lightgreen",
            edgecolor="green",
        )

        ax.grid(True, linestyle="--", alpha=0.7, axis="y")
        ax.set_title(
            f"{metric_label} Comparison for {search_type.capitalize()} Search", pad=20
        )
        ax.set_xlabel("Concurrent Queries", labelpad=10)
        ax.set_ylabel(metric_label, labelpad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend(fontsize=10)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax.annotate(
                        f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=8,
                    )

        autolabel(postgres_bars)
        autolabel(mongodb_bars)

        plt.tight_layout()
        plt.show()

bar_chart_performance_analysis_comparison(
    performance_guidance_results,
    metric="avg_latency",
    metric_label="Average Latency (ms)",
)
bar_chart_performance_analysis_comparison(
    performance_guidance_results,
    metric="throughput",
    metric_label="Throughput (queries/sec)",
)
bar_chart_performance_analysis_comparison(
    performance_guidance_results, metric="p95_latency", metric_label="P95 Latency (ms)"
)

"""
## Part 5: Extra Notes

### 5.1 PostgreSQL JSONB vs MongoDB BSON

| Feature                    | **PostgreSQL JSONB**                              | **MongoDB BSON**                              |
|----------------------------|--------------------------------------------------|----------------------------------------------|
| **Integration**            | An extension to a relational database system.    | Native to MongoDB, a document database.      |
| **Query Language**         | Uses SQL with JSONB-specific operators/functions. | Uses MongoDB Query Language (MQL), a JSON-like query syntax. |
| **Storage Optimization**   | Optimized for relational data alongside JSONB.   | Fully optimized for JSON-like document storage. |
| **Data Type Support**      | Stores standard JSON data types (e.g., strings, numbers). | Includes additional types not in standard JSON (e.g., `Date`, `ObjectId`, `Binary`). |
| **Use Case**               | Best for hybrid relational/JSON use cases.       | Designed for flexible schemas, document-based databases. |
| **Updates**                | JSONB supports in-place updates for specific keys or paths. | BSON supports in-place updates with more native support for field-level atomic operations. |
| **Size Overhead**          | Slightly more compact than BSON in some cases.   | Includes metadata like type information, leading to slightly larger size. |

### 5.2 Limitations of pgvector for Handling Large-Dimensional Embeddings

While pgvector is a powerful tool for storing and searching vector embeddings in PostgreSQL, it does have inherent limitations when it comes to handling very high-dimensional embeddings. Here are the key points and [source](https://github.com/pgvector/pgvector/issues/461):

- **PostgreSQL Page Size Constraint:**  
  - **Reason:** PostgreSQL uses fixed 8KB pages for data storage.  
  - **Impact:** Each 32-bit float occupies 4 bytes, so storing a vector with many dimensions quickly exhausts the available space on a page.  
  - **Practical Limit:** This design limits indexed vectors to around 2000 dimensions unless alternative approaches (such as quantization or splitting the vector) are used.

- **Index Tuple Size Limit:**  
  - **Reason:** Even if the underlying table supports larger vectors (up to 16,000 dimensions), the index tuples themselves are constrained by the 8KB limit.  
  - **Impact:** Attempting to build an index on vectors exceeding this limit results in errors or performance degradation.

- **Trade-offs in Workarounds:**  
  - **Quantization:** Converting vectors from 32-bit floats to lower precision (e.g., half-precision) can allow for more dimensions but may reduce accuracy.  
  - **Splitting Vectors:** Dividing a high-dimensional vector across multiple columns or rows increases complexity in reconstructing the original vector for search and may affect retrieval speed.
  - **Alternative Data Types:** Some projects (like pgvecto.rs) bypass these limitations by handling indexing outside PostgreSQL, but this sacrifices the ACID guarantees that pgvector provides.

- **Implications for AI Workloads:**  
  - **Model Compatibility:** Many modern embedding models (e.g., Ollama’s `text-embedding-3-large` with 3072 dimensions) produce embeddings that exceed pgvector’s optimal indexed dimension size, potentially forcing truncation or quantization.
  - **Search Quality:** These workarounds (truncation, quantization, or splitting) can impact the precision and recall of similarity searches—a critical factor for many AI applications.

### 5.3 Workaround Options for High-Dimensional Embeddings in pgvector

The pgvector project has received several suggestions and workaround proposals to mitigate the limitation of indexing high-dimensional vectors (beyond ~2000 dimensions). Two key comments from issues [#326](https://github.com/pgvector/pgvector/issues/326#issuecomment-2024106976) and [#395](https://github.com/pgvector/pgvector/issues/395#issuecomment-2024089498) detail some of these options.

---

#### 5.3.1 Option 1: Use Lower Precision with `halfvec`

- **Description:**  
  Convert full-precision (fp32) vectors to half-precision (fp16) for indexing purposes.
  
- **How It Works:**  
  - **Storage:** Vectors are stored as `vector(n)` (still in fp32) in the table.
  - **Indexing:** When creating the index, the vector is cast to the `halfvec` type (fp16) using syntax like:
    ```sql
    CREATE INDEX ON items USING hnsw ((embedding::halfvec(n)) halfvec_l2_ops);
    ```
  
- **Benefits:**  
  - **Smaller Index Size:** fp16 values require half the storage of fp32, which can allow more dimensions to fit within the 8KB index tuple limit.
  - **Faster Index Build:** Smaller data size can lead to quicker index creation.
  
- **Trade-offs:**  
  - **Loss of Precision:** Quantizing from 32-bit to 16-bit floats introduces rounding errors.  
  - **Impact on Recall:** Testing has shown that recall remains nearly identical in many cases, but the loss in precision may not be acceptable for all applications.

- **Appropriateness:**  
  This workaround is appropriate if the application can tolerate a slight reduction in numerical precision without significantly affecting the quality of similarity search results.

---

#### 5.3.2 Option 2: Split the Embedding Across Multiple Rows or Columns

- **Description:**  
  Divide a high-dimensional vector into multiple smaller vectors that can be stored and indexed separately.
  
- **How It Works:**  
  - **Schema Change:** Instead of storing one vector with dimensions greater than the limit, split it into two or more parts (e.g., a 3072-dimensional vector into one part of 2000 dimensions and another of 1072 dimensions).
  - **Indexing:** Build separate indexes for each part and then combine the results (possibly with a re-ranking step) during query time.
  
- **Benefits:**  
  - **Full Precision Retained:** No need to quantize the data, so the original accuracy is preserved.
  - **Scalability:** This approach can support arbitrarily high dimensions by splitting the data.
  
- **Trade-offs:**  
  - **Complexity:** Requires changes to the schema and additional logic in query processing to recombine or re-rank partial results.
  - **Performance Overhead:** Merging results from multiple indexes can add latency to the search process.

- **Appropriateness:**  
  This workaround is most appropriate when high precision is critical and the application cannot afford any loss in accuracy. It adds complexity but retains full precision for each vector.
"""
logger.info("## Part 5: Extra Notes")

logger.info("\n\n[DONE]", bright=True)