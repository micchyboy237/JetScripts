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
# Salesforce

[Salesforce](https://www.salesforce.com/) is a cloud-based software company that
provides customer relationship management (CRM) solutions and a suite of enterprise
applications focused on sales, customer service, marketing automation, and analytics.

[langchain-salesforce](https://pypi.org/project/langchain-salesforce/) implements
tools enabling LLMs to interact with Salesforce data.


## Installation and Setup
"""
logger.info("# Salesforce")

pip install langchain-salesforce

"""
## Tools

See detail on available tools [here](/docs/integrations/tools/salesforce/).
"""
logger.info("## Tools")

logger.info("\n\n[DONE]", bright=True)