from jet.logger import logger
from langchain_community.vectorstores.nucliadb import NucliaDB
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
# NucliaDB

You can use a local NucliaDB instance or use [Nuclia Cloud](https://nuclia.cloud).

When using a local instance, you need a Nuclia Understanding API key, so your texts are properly vectorized and indexed. You can get a key by creating a free account at [https://nuclia.cloud](https://nuclia.cloud), and then [create a NUA key](https://docs.nuclia.dev/docs/docs/using/understanding/intro).
"""
logger.info("# NucliaDB")

# %pip install --upgrade --quiet  langchain langchain-community nuclia

"""
## Usage with nuclia.cloud
"""
logger.info("## Usage with nuclia.cloud")


API_KEY = "YOUR_API_KEY"

ndb = NucliaDB(knowledge_box="YOUR_KB_ID", local=False, api_key=API_KEY)

"""
## Usage with a local instance

Note: By default `backend` is set to `http://localhost:8080`.
"""
logger.info("## Usage with a local instance")


ndb = NucliaDB(knowledge_box="YOUR_KB_ID", local=True, backend="http://my-local-server")

"""
## Add and delete texts to your Knowledge Box
"""
logger.info("## Add and delete texts to your Knowledge Box")

ids = ndb.add_texts(["This is a new test", "This is a second test"])

ndb.delete(ids=ids)

"""
## Search in your Knowledge Box
"""
logger.info("## Search in your Knowledge Box")

results = ndb.similarity_search("Who was inspired by Ada Lovelace?")
logger.debug(results[0].page_content)

logger.info("\n\n[DONE]", bright=True)