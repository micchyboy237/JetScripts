from jet.logger import logger
from langchain_community.embeddings import LLMRailsEmbeddings
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
# LLMRails

Let's load the LLMRails Embeddings class.

To use LLMRails embedding you need to pass api key by argument or set it in environment with `LLM_RAILS_API_KEY` key.
To gey API Key you need to sign up in https://console.llmrails.com/signup and then go to https://console.llmrails.com/api-keys and copy key from there after creating one key in platform.
"""
logger.info("# LLMRails")


embeddings = LLMRailsEmbeddings(model="embedding-english-v1")  # or embedding-multi-v1

text = "This is a test document."

"""
To generate embeddings, you can either query an invidivual text, or you can query a list of texts.
"""
logger.info("To generate embeddings, you can either query an invidivual text, or you can query a list of texts.")

query_result = embeddings.embed_query(text)
query_result[:5]

doc_result = embeddings.embed_documents([text])
doc_result[0][:5]

logger.info("\n\n[DONE]", bright=True)