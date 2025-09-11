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
# Featherless AI

[Featherless AI](https://featherless.ai/) is a serverless AI inference platform that offers access to over 4300+ open-source models. Our goal is to make all AI models available for serverless inference. We provide inference via API to a continually expanding library of open-weight models.

# Installation and Setup
`pip install langchain-featherless-ai`
1. Sign up for an account at [Featherless](https://featherless.ai/register)
2. Subscribe to a plan and get your API key from [API Keys](https://featherless.ai/account/api-keys)
3. Set up your API key as an environment variable(`FEATHERLESSAI_API_KEY`)

# Model catalog
Visit our model catalog for an overview of all our models: https://featherless.ai/models
"""
logger.info("# Featherless AI")

logger.info("\n\n[DONE]", bright=True)