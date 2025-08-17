from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Overview
icon: "info"
iconType: "solid"
---

Mem0 includes built-in support for various popular databases. Memory can utilize the database provided by the user, ensuring efficient use for specific needs.

## Supported Vector Databases

See the list of supported vector databases below.

<Note>
  The following vector databases are supported in the Python implementation. The TypeScript implementation currently only supports Qdrant, Redis,Vectorize and in-memory vector database.
</Note>

<CardGroup cols={3}>
  <Card title="Qdrant" href="/components/vectordbs/dbs/qdrant"></Card>
  <Card title="Chroma" href="/components/vectordbs/dbs/chroma"></Card>
  <Card title="Pgvector" href="/components/vectordbs/dbs/pgvector"></Card>
  <Card title="Upstash Vector" href="/components/vectordbs/dbs/upstash-vector"></Card>
  <Card title="Milvus" href="/components/vectordbs/dbs/milvus"></Card>
  <Card title="Pinecone" href="/components/vectordbs/dbs/pinecone"></Card>
  <Card title="MongoDB" href="/components/vectordbs/dbs/mongodb"></Card>
  <Card title="Azure" href="/components/vectordbs/dbs/azure"></Card>
  <Card title="Redis" href="/components/vectordbs/dbs/redis"></Card>
  <Card title="Elasticsearch" href="/components/vectordbs/dbs/elasticsearch"></Card>
  <Card title="OpenSearch" href="/components/vectordbs/dbs/opensearch"></Card>
  <Card title="Supabase" href="/components/vectordbs/dbs/supabase"></Card>
  <Card title="Vertex AI" href="/components/vectordbs/dbs/vertex_ai"></Card>
  <Card title="Weaviate" href="/components/vectordbs/dbs/weaviate"></Card>
  <Card title="FAISS" href="/components/vectordbs/dbs/faiss"></Card>
  <Card title="LangChain" href="/components/vectordbs/dbs/langchain"></Card>
</CardGroup>

## Usage

To utilize a vector database, you must provide a configuration to customize its usage. If no configuration is supplied, a default configuration will be applied, and `Qdrant` will be used as the vector database.

For a comprehensive list of available parameters for vector database configuration, please refer to [Config](./config).

## Common issues

### Using model with different dimensions

If you are using customized model, which is having different dimensions other than 1536
for example 768, you may encounter below error:

`ValueError: shapes (0,1536) and (768,) not aligned: 1536 (dim 1) != 768 (dim 0)`

you could add `"embedding_model_dims": 768,` to the config of the vector_store to overcome this issue.
"""
logger.info("## Supported Vector Databases")

logger.info("\n\n[DONE]", bright=True)