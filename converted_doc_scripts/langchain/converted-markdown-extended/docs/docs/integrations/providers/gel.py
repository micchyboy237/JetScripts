from jet.logger import logger
from langchain_gel import GelVectorStore
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
# Gel

[Gel](https://www.geldata.com/) is a powerful data platform built on top of PostgreSQL.

- Think in objects and graphs instead of tables and JOINs.
- Use the advanced Python SDK, integrated GUI, migrations engine, Auth and AI layers, and much more.
- Run locally, remotely, or in a [fully managed cloud](https://www.geldata.com/cloud).

## Installation
"""
logger.info("# Gel")

pip install langchain-gel

"""
## Setup

1. Run `gel project init`
2. Edit the schema. You need the following types to use the LangChain vectorstore:
"""
logger.info("## Setup")

using extension pgvector;

module default {
    scalar type EmbeddingVector extending ext::pgvector::vector<1536>;

    type Record {
        required collection: str;
        text: str;
        embedding: EmbeddingVector;
        external_id: str {
            constraint exclusive;
        };
        metadata: json;

        index ext::pgvector::hnsw_cosine(m := 16, ef_construction := 128)
            on (.embedding)
    }
}

"""
> Note: this is the minimal setup. Feel free to add as many types, properties and links as you want!
> Learn more about taking advantage of Gel's schema by reading the [docs](https://docs.geldata.com/learn/schema).

3. Run the migration: `gel migration create && gel migrate`.

## Usage
"""
logger.info("## Usage")


vector_store = GelVectorStore(
    embeddings=embeddings,
)

"""
See the full usage example [here](/docs/integrations/vectorstores/gel).
"""
logger.info("See the full usage example [here](/docs/integrations/vectorstores/gel).")

logger.info("\n\n[DONE]", bright=True)