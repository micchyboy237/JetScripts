from jet.logger import logger
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
# Smabbler
> Smabbler’s graph-powered platform boosts AI development by transforming data into a structured knowledge foundation.

# Galaxia

> Galaxia Knowledge Base is an integrated knowledge base and retrieval mechanism for RAG. In contrast to standard solution, it is based on Knowledge Graphs built using symbolic NLP and Knowledge Representation solutions. Provided texts are analysed and transformed into Graphs containing text, language and semantic information. This rich structure allows for retrieval that is based on semantic information, not on vector similarity/distance.

Implementing RAG using Galaxia involves first uploading your files to [Galaxia](https://beta.cloud.smabbler.com/home), analyzing them there and then building a model (knowledge graph). When the model is built, you can use `GalaxiaRetriever` to connect to the API and start retrieving.

More information: [docs](https://smabbler.gitbook.io/smabbler)

## Installation

# pip install langchain-galaxia-retriever

## Usage
"""
logger.info("# Smabbler")

logger.info("\n\n[DONE]", bright=True)