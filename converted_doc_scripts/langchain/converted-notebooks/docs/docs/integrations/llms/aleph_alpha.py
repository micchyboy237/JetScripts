from jet.logger import logger
from langchain_community.llms import AlephAlpha
from langchain_core.prompts import PromptTemplate
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
# Aleph Alpha

[The Luminous series](https://docs.aleph-alpha.com/docs/category/luminous/) is a family of large language models.

This example goes over how to use LangChain to interact with Aleph Alpha models
"""
logger.info("# Aleph Alpha")

# %pip install -qU langchain-community

# %pip install --upgrade --quiet  aleph-alpha-client

# from getpass import getpass

# ALEPH_ALPHA_API_KEY = getpass()


template = """Q: {question}

A:"""

prompt = PromptTemplate.from_template(template)

llm = AlephAlpha(
    model="luminous-extended",
    maximum_tokens=20,
    stop_sequences=["Q:"],
    aleph_alpha_api_key=ALEPH_ALPHA_API_KEY,
)

llm_chain = prompt | llm

question = "What is AI?"

llm_chain.invoke({"question": question})

logger.info("\n\n[DONE]", bright=True)