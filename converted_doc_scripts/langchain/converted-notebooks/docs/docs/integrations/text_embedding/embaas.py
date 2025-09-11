from jet.logger import logger
from langchain_community.embeddings import EmbaasEmbeddings
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
# Embaas

[embaas](https://embaas.io) is a fully managed NLP API service that offers features like embedding generation, document text extraction, document to embeddings and more. You can choose a [variety of pre-trained models](https://embaas.io/docs/models/embeddings).

In this tutorial, we will show you how to use the embaas Embeddings API to generate embeddings for a given text.

### Prerequisites
Create your free embaas account at [https://embaas.io/register](https://embaas.io/register) and generate an [API key](https://embaas.io/dashboard/api-keys).
"""
logger.info("# Embaas")


embaas_
os.environ["EMBAAS_API_KEY"] = "YOUR_API_KEY"


embeddings = EmbaasEmbeddings()

doc_text = "This is a test document."
doc_text_embedding = embeddings.embed_query(doc_text)

logger.debug(doc_text_embedding)

doc_texts = ["This is a test document.", "This is another test document."]
doc_texts_embeddings = embeddings.embed_documents(doc_texts)

for i, doc_text_embedding in enumerate(doc_texts_embeddings):
    logger.debug(f"Embedding for document {i + 1}: {doc_text_embedding}")

embeddings = EmbaasEmbeddings(
    model="instructor-large",
    instruction="Represent the Wikipedia document for retrieval",
)

"""
For more detailed information about the embaas Embeddings API, please refer to [the official embaas API documentation](https://embaas.io/api-reference).
"""
logger.info("For more detailed information about the embaas Embeddings API, please refer to [the official embaas API documentation](https://embaas.io/api-reference).")

logger.info("\n\n[DONE]", bright=True)