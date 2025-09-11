from jet.logger import logger
from langchain_community.document_loaders import AirbyteJSONLoader
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
---
sidebar_class_name: hidden
---

# Airbyte JSON (Deprecated)

Note: `AirbyteJSONLoader` is deprecated. Please use [`AirbyteLoader`](/docs/integrations/document_loaders/airbyte) instead.

>[Airbyte](https://github.com/airbytehq/airbyte) is a data integration platform for ELT pipelines from APIs, databases & files to warehouses & lakes. It has the largest catalog of ELT connectors to data warehouses and databases.

This covers how to load any source from Airbyte into a local JSON file that can be read in as a document

Prereqs:
Have docker desktop installed

Steps:

1) Clone Airbyte from GitHub - `git clone https://github.com/airbytehq/airbyte.git`

2) Switch into Airbyte directory - `cd airbyte`

3) Start Airbyte - `docker compose up`

4) In your browser, just visit http://localhost:8000. You will be asked for a username and password. By default, that's username `airbyte` and password `password`.

5) Setup any source you wish.

6) Set destination as Local JSON, with specified destination path - let's say `/json_data`. Set up manual sync.

7) Run the connection.

7) To see what files are created, you can navigate to: `file:///tmp/airbyte_local`

8) Find your data and copy path. That path should be saved in the file variable below. It should start with `/tmp/airbyte_local`
"""
logger.info("# Airbyte JSON (Deprecated)")


# !ls /tmp/airbyte_local/json_data/

loader = AirbyteJSONLoader("/tmp/airbyte_local/json_data/_airbyte_raw_pokemon.jsonl")

data = loader.load()

logger.debug(data[0].page_content[:500])

logger.info("\n\n[DONE]", bright=True)