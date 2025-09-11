from jet.logger import logger
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders.larksuite import (
LarkSuiteDocLoader,
LarkSuiteWikiLoader,
)
from langchain_community.llms.fake import FakeListLLM
from pprint import pprint
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
# LarkSuite (FeiShu)

>[LarkSuite](https://www.larksuite.com/) is an enterprise collaboration platform developed by ByteDance.

This notebook covers how to load data from the `LarkSuite` REST API into a format that can be ingested into LangChain, along with example usage for text summarization.

The LarkSuite API requires an access token (tenant_access_token or user_access_token), checkout [LarkSuite open platform document](https://open.larksuite.com/document) for API details.
"""
logger.info("# LarkSuite (FeiShu)")

# from getpass import getpass


DOMAIN = input("larksuite domain")
# ACCESS_TOKEN = getpass("larksuite tenant_access_token or user_access_token")
DOCUMENT_ID = input("larksuite document id")

"""
## Load From Document
"""
logger.info("## Load From Document")


larksuite_loader = LarkSuiteDocLoader(DOMAIN, ACCESS_TOKEN, DOCUMENT_ID)
docs = larksuite_loader.load()

plogger.debug(docs)

"""
## Load From Wiki
"""
logger.info("## Load From Wiki")


DOCUMENT_ID = input("larksuite wiki id")
larksuite_loader = LarkSuiteWikiLoader(DOMAIN, ACCESS_TOKEN, DOCUMENT_ID)
docs = larksuite_loader.load()

plogger.debug(docs)


llm = FakeListLLM()
chain = load_summarize_chain(llm, chain_type="map_reduce")
chain.run(docs)

logger.info("\n\n[DONE]", bright=True)