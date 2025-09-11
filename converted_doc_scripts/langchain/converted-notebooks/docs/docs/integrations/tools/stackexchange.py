from jet.logger import logger
from langchain_community.utilities import StackExchangeAPIWrapper
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
# StackExchange

>[Stack Exchange](https://stackexchange.com/) is a network of question-and-answer (Q&A) websites on topics in diverse fields, each site covering a specific topic, where questions, answers, and users are subject to a reputation award process. The reputation system allows the sites to be self-moderating.

The ``StackExchange`` component integrates the StackExchange API into LangChain allowing access to the [StackOverflow](https://stackoverflow.com/) site of the Stack Excchange network. Stack Overflow focuses on computer programming.


This notebook goes over how to use the ``StackExchange`` component.

We first have to install the python package stackapi which implements the Stack Exchange API.
"""
logger.info("# StackExchange")

pip install --upgrade stackapi


stackexchange = StackExchangeAPIWrapper()

stackexchange.run("zsh: command not found: python")

logger.info("\n\n[DONE]", bright=True)