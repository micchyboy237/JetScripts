from jet.logger import logger
from langchain_community.llms import KoboldApiLLM
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
# KoboldAI API

[KoboldAI](https://github.com/KoboldAI/KoboldAI-Client) is a "a browser-based front-end for AI-assisted writing with multiple local & remote AI models...". It has a public and local API that is able to be used in langchain.

This example goes over how to use LangChain with that API.

Documentation can be found in the browser adding /api to the end of your endpoint (i.e http://127.0.0.1/:5000/api).
"""
logger.info("# KoboldAI API")


"""
Replace the endpoint seen below with the one shown in the output after starting the webui with --api or --public-api

Optionally, you can pass in parameters like temperature or max_length
"""
logger.info("Replace the endpoint seen below with the one shown in the output after starting the webui with --api or --public-api")

llm = KoboldApiLLM(endpoint="http://192.168.1.144:5000", max_length=80)

response = llm.invoke(
    "### Instruction:\nWhat is the first book of the bible?\n### Response:"
)

logger.info("\n\n[DONE]", bright=True)