from jet.logger import logger
from langchain_community.agent_toolkits.clickup.toolkit import ClickupToolkit
from langchain_community.utilities.clickup import ClickupAPIWrapper
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
# ClickUp

>[ClickUp](https://clickup.com/) is an all-in-one productivity platform that provides small and large teams across industries with flexible and customizable work management solutions, tools, and functions.
>
>It is a cloud-based project management solution for businesses of all sizes featuring communication and collaboration tools to help achieve organizational goals.

## Installation and Setup

1. Create a [ClickUp App](https://help.clickup.com/hc/en-us/articles/6303422883095-Create-your-own-app-with-the-ClickUp-API)
2. Follow [these steps](https://clickup.com/api/developer-portal/authentication/) to get your client_id and client_secret.

## Toolkits
"""
logger.info("# ClickUp")


"""
See a [usage example](/docs/integrations/tools/clickup).
"""
logger.info("See a [usage example](/docs/integrations/tools/clickup).")

logger.info("\n\n[DONE]", bright=True)