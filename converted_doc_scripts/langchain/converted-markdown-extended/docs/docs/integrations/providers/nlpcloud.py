from jet.logger import logger
from langchain_community.embeddings import NLPCloudEmbeddings
from langchain_community.llms import NLPCloud
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
# NLPCloud

>[NLP Cloud](https://docs.nlpcloud.com/#introduction) is an artificial intelligence platform that allows you to use the most advanced AI engines, and even train your own engines with your own data.


## Installation and Setup

- Install the `nlpcloud` package.
"""
logger.info("# NLPCloud")

pip install nlpcloud

"""
- Get an NLPCloud api key and set it as an environment variable (`NLPCLOUD_API_KEY`)


## LLM

See a [usage example](/docs/integrations/llms/nlpcloud).
"""
logger.info("## LLM")


"""
## Text Embedding Models

See a [usage example](/docs/integrations/text_embedding/nlp_cloud)
"""
logger.info("## Text Embedding Models")


logger.info("\n\n[DONE]", bright=True)