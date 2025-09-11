from jet.logger import logger
from langchain_hana import HanaDB
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
# SAP

>[SAP SE(Wikipedia)](https://www.sap.com/about/company.html) is a German multinational
> software company. It develops enterprise software to manage business operation and
> customer relations. The company is the world's leading
> `enterprise resource planning (ERP)` software vendor.

## Installation and Setup

We need to install the `langchain-hana` python package.
"""
logger.info("# SAP")

pip install langchain-hana

"""
## Vectorstore

>[SAP HANA Cloud Vector Engine](https://help.sap.com/docs/hana-cloud-database/sap-hana-cloud-sap-hana-database-vector-engine-guide/sap-hana-cloud-sap-hana-database-vector-engine-guide) is
> a vector store fully integrated into the `SAP HANA Cloud` database.

See a [usage example](/docs/integrations/vectorstores/sap_hanavector).
"""
logger.info("## Vectorstore")


logger.info("\n\n[DONE]", bright=True)