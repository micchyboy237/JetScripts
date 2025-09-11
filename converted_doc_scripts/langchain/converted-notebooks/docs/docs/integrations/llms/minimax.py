from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.llms import Minimax
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
# Minimax

[Minimax](https://api.minimax.chat) is a Chinese startup that provides natural language processing models for companies and individuals.

This example demonstrates using Langchain to interact with Minimax.

# Setup

To run this notebook, you'll need a [Minimax account](https://api.minimax.chat), an [API key](https://api.minimax.chat/user-center/basic-information/interface-key), and a [Group ID](https://api.minimax.chat/user-center/basic-information)

# Single model call
"""
logger.info("# Minimax")


minimax = Minimax(minimax_minimax_group_id="YOUR_GROUP_ID")

minimax("What is the difference between panda and bear?")

"""
# Chained model calls
"""
logger.info("# Chained model calls")


os.environ["MINIMAX_API_KEY"] = "YOUR_API_KEY"
os.environ["MINIMAX_GROUP_ID"] = "YOUR_GROUP_ID"


template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

llm = Minimax()

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NBA team won the Championship in the year Jay Zhou was born?"

llm_chain.run(question)

logger.info("\n\n[DONE]", bright=True)