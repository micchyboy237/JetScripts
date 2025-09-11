from jet.logger import logger
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
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
# SpaCy

>[spaCy](https://spacy.io/) is an open-source software library for advanced natural language processing, written in the programming languages Python and Cython.
 

## Installation and Setup
"""
logger.info("# SpaCy")

# %pip install --upgrade --quiet  spacy

"""
Import the necessary classes
"""
logger.info("Import the necessary classes")


"""
## Example

Initialize SpacyEmbeddings.This will load the Spacy model into memory.
"""
logger.info("## Example")

embedder = SpacyEmbeddings(model_name="en_core_web_sm")

"""
Define some example texts . These could be any documents that you want to analyze - for example, news articles, social media posts, or product reviews.
"""
logger.info("Define some example texts . These could be any documents that you want to analyze - for example, news articles, social media posts, or product reviews.")

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump!",
    "Bright vixens jump; dozy fowl quack.",
]

"""
Generate and print embeddings for the texts . The SpacyEmbeddings class generates an embedding for each document, which is a numerical representation of the document's content. These embeddings can be used for various natural language processing tasks, such as document similarity comparison or text classification.
"""
logger.info("Generate and print embeddings for the texts . The SpacyEmbeddings class generates an embedding for each document, which is a numerical representation of the document's content. These embeddings can be used for various natural language processing tasks, such as document similarity comparison or text classification.")

embeddings = embedder.embed_documents(texts)
for i, embedding in enumerate(embeddings):
    logger.debug(f"Embedding for document {i + 1}: {embedding}")

"""
Generate and print an embedding for a single piece of text. You can also generate an embedding for a single piece of text, such as a search query. This can be useful for tasks like information retrieval, where you want to find documents that are similar to a given query.
"""
logger.info("Generate and print an embedding for a single piece of text. You can also generate an embedding for a single piece of text, such as a search query. This can be useful for tasks like information retrieval, where you want to find documents that are similar to a given query.")

query = "Quick foxes and lazy dogs."
query_embedding = embedder.embed_query(query)
logger.debug(f"Embedding for query: {query_embedding}")

logger.info("\n\n[DONE]", bright=True)