from jet.logger import logger
from tilores import TiloresAPI
from tilores_langchain import TiloresTools
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
# Tilores

[Tilores](https://tilores.io) is a platform that provides advanced entity resolution solutions for data integration and management. Using cutting-edge algorithms, machine learning, and a user-friendly interfaces, Tilores helps organizations match, resolve, and consolidate data from disparate sources, ensuring high-quality, consistent information.

## Installation and Setup
"""
logger.info("# Tilores")

# %pip install --upgrade tilores-langchain

"""
To access Tilores, you need to [create and configure an instance](https://app.tilores.io). If you prefer to test out Tilores first, you can use the [read-only demo credentials](https://github.com/tilotech/identity-rag-customer-insights-chatbot?tab=readme-ov-file#1-configure-customer-data-access).
"""
logger.info("To access Tilores, you need to [create and configure an instance](https://app.tilores.io). If you prefer to test out Tilores first, you can use the [read-only demo credentials](https://github.com/tilotech/identity-rag-customer-insights-chatbot?tab=readme-ov-file#1-configure-customer-data-access).")



os.environ["TILORES_API_URL"] = "<api-url>"
os.environ["TILORES_TOKEN_URL"] = "<token-url>"
os.environ["TILORES_CLIENT_ID"] = "<client-id>"
os.environ["TILORES_CLIENT_SECRET"] = "<client-secret>"

tilores = TiloresAPI.from_environ()

"""
Please refer to the [Tilores documentation](https://docs.tilotech.io/tilores/publicsaaswalkthrough/) on how to create your own instance.

## Toolkits

You can use the [`TiloresTools`](/docs/integrations/tools/tilores) to query data from Tilores:
"""
logger.info("## Toolkits")


logger.info("\n\n[DONE]", bright=True)