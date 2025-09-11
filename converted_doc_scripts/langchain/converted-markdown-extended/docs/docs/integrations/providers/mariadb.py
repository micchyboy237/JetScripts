from jet.logger import logger
from langchain_mariadb import MariaDBStore
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
# MariaDB

This page covers how to use the [MariaDB](https://github.com/mariadb/) ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific PGVector wrappers.

## Installation
- Install c/c connector

on Debian, Ubuntu
"""
logger.info("# MariaDB")

sudo apt install libmariadb3 libmariadb-dev

"""
on CentOS, RHEL, Rocky Linux
"""
logger.info("on CentOS, RHEL, Rocky Linux")

sudo yum install MariaDB-shared MariaDB-devel

"""
- Install the Python connector package with `pip install mariadb`


## Setup
1. The first step is to have a MariaDB 11.7.1 or later installed.

    The docker image is the easiest way to get started.

## Wrappers

### VectorStore

There exists a wrapper around MariaDB vector databases, allowing you to use it as a vectorstore,
whether for semantic search or example selection.

To import this vectorstore:
"""
logger.info("## Setup")


"""
### Usage

For a more detailed walkthrough of the MariaDB wrapper, see [this notebook](/docs/integrations/vectorstores/mariadb)
"""
logger.info("### Usage")

logger.info("\n\n[DONE]", bright=True)