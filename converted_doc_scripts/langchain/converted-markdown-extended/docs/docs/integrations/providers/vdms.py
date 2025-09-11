from jet.logger import logger
from langchain_vdms import VDMS
from langchain_vdms.vectorstores import VDMS
from langchain_vdms.vectorstores import VDMS_Client
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
# VDMS

> [VDMS](https://github.com/IntelLabs/vdms/blob/master/README.md) is a storage solution for efficient access
> of big-”visual”-data that aims to achieve cloud scale by searching for relevant visual data via visual metadata
> stored as a graph and enabling machine friendly enhancements to visual data for faster access.

## Installation and Setup

### Install Client
"""
logger.info("# VDMS")

pip install langchain-vdms

"""
### Install Database

There are two ways to get started with VDMS:


1. Install VDMS on your local machine via docker
    ```bash
        docker run -d -p 55555:55555 intellabs/vdms:latest
    ```

2. Install VDMS directly on your local machine. Please see
[installation instructions](https://github.com/IntelLabs/vdms/blob/master/INSTALL.md).

## VectorStore

To import this vectorstore:
"""
logger.info("### Install Database")


"""
To import the VDMS Client connector:
"""
logger.info("To import the VDMS Client connector:")


"""
For a more detailed walkthrough of the VDMS wrapper, see [this guide](/docs/integrations/vectorstores/vdms).
"""
logger.info("For a more detailed walkthrough of the VDMS wrapper, see [this guide](/docs/integrations/vectorstores/vdms).")

logger.info("\n\n[DONE]", bright=True)