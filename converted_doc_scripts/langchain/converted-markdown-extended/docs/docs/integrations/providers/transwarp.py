from jet.logger import logger
from langchain_community.vectorstores.hippo import Hippo
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
# Transwarp

>[Transwarp](https://www.transwarp.cn/en/introduction) aims to build
> enterprise-level big data and AI infrastructure software,
> to shape the future of data world. It provides enterprises with
> infrastructure software and services around the whole data lifecycle,
> including integration, storage, governance, modeling, analysis,
> mining and circulation.
>
> `Transwarp` focuses on technology research and
> development and has accumulated core technologies in these aspects:
> distributed computing, SQL compilations, database technology,
> unification for multi-model data management, container-based cloud computing,
> and big data analytics and intelligence.

## Installation

You have to install several python packages:
"""
logger.info("# Transwarp")

pip install -U tiktoken hippo-api

"""
and get the connection configuration.

## Vector stores

### Hippo

See [a usage example and installation instructions](/docs/integrations/vectorstores/hippo).
"""
logger.info("## Vector stores")


logger.info("\n\n[DONE]", bright=True)