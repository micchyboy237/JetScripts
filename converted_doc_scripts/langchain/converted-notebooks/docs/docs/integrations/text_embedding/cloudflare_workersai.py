from dotenv import load_dotenv
from jet.logger import logger
from langchain_cloudflare.embeddings import (
CloudflareWorkersAIEmbeddings,
)
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
# Cloudflare Workers AI

>[Cloudflare, Inc. (Wikipedia)](https://en.wikipedia.org/wiki/Cloudflare) is an American company that provides content delivery network services, cloud cybersecurity, DDoS mitigation, and ICANN-accredited domain registration services.

>[Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/) allows you to run machine learning models, on the `Cloudflare` network, from your code via REST API.

>[Workers AI Developer Docs](https://developers.cloudflare.com/workers-ai/models/text-embeddings/) lists all text embeddings models available.

## Setting up

Both a Cloudflare Account ID and Workers AI API token are required. Find how to obtain them from [this document](https://developers.cloudflare.com/workers-ai/get-started/rest-api/).

You can pass these parameters explicitly or define as environmental variables.
"""
logger.info("# Cloudflare Workers AI")



load_dotenv(".env")

cf_acct_id = os.getenv("CF_ACCOUNT_ID")

cf_ai_token = os.getenv("CF_AI_API_TOKEN")

"""
## Example
"""
logger.info("## Example")


embeddings = CloudflareWorkersAIEmbeddings(
    account_id=cf_acct_id,
    api_token=cf_ai_token,
    model_name="@cf/baai/bge-small-en-v1.5",
)
query_result = embeddings.embed_query("test")
len(query_result), query_result[:3]

batch_query_result = embeddings.embed_documents(["test1", "test2", "test3"])
len(batch_query_result), len(batch_query_result[0])

logger.info("\n\n[DONE]", bright=True)