from jet.logger import logger
from langchain_cloudflare import ChatCloudflareWorkersAI
from langchain_cloudflare import CloudflareVectorize
from langchain_cloudflare import CloudflareWorkersAIEmbeddings
from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI
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
# Cloudflare

>[Cloudflare, Inc. (Wikipedia)](https://en.wikipedia.org/wiki/Cloudflare) is an American company that provides
> content delivery network services, cloud cybersecurity, DDoS mitigation, and ICANN-accredited
> domain registration services.

>[Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/) allows you to run machine
> learning models, on the `Cloudflare` network, from your code via REST API.


## ChatModels

See [installation instructions and usage example](/docs/integrations/chat/cloudflare_workersai).
"""
logger.info("# Cloudflare")


"""
## VectorStore

See [installation instructions and usage example](/docs/integrations/vectorstores/cloudflare_vectorize).
"""
logger.info("## VectorStore")


"""
## Embeddings

See [installation instructions and usage example](/docs/integrations/text_embedding/cloudflare_workersai).
"""
logger.info("## Embeddings")


"""
## LLMs

See [installation instructions and usage example](/docs/integrations/llms/cloudflare_workersai).
"""
logger.info("## LLMs")


logger.info("\n\n[DONE]", bright=True)