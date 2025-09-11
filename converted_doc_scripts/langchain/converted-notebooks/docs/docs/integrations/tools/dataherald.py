from jet.logger import logger
from langchain_community.utilities.dataherald import DataheraldAPIWrapper
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
# Dataherald

This notebook goes over how to use the dataherald component.

First, you need to set up your Dataherald account and get your API KEY:

1. Go to dataherald and sign up [here](https://www.dataherald.com/)
2. Once you are logged in your Admin Console, create an API KEY
3. pip install dataherald

Then we will need to set some environment variables:
1. Save your API KEY into DATAHERALD_API_KEY env variable
"""
logger.info("# Dataherald")

pip install dataherald
# %pip install --upgrade --quiet langchain-community


os.environ["DATAHERALD_API_KEY"] = ""


dataherald = DataheraldAPIWrapper(db_connection_id="65fb766367dd22c99ce1a12d")

dataherald.run("How many employees are in the company?")

logger.info("\n\n[DONE]", bright=True)