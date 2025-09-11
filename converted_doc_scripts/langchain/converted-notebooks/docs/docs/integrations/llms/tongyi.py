from jet.logger import logger
from langchain_community.llms import Tongyi
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
# Tongyi Qwen
Tongyi Qwen is a large-scale language model developed by Alibaba's Damo Academy. It is capable of understanding user intent through natural language understanding and semantic analysis, based on user input in natural language. It provides services and assistance to users in different domains and tasks. By providing clear and detailed instructions, you can obtain results that better align with your expectations.

## Setting up
"""
logger.info("# Tongyi Qwen")

# %pip install --upgrade --quiet  langchain-community dashscope

# from getpass import getpass

# DASHSCOPE_API_KEY = getpass()


os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY


Tongyi().invoke("What NFL team won the Super Bowl in the year Justin Bieber was born?")

"""
## Using in a chain
"""
logger.info("## Using in a chain")


llm = Tongyi()

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

chain = prompt | llm

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

chain.invoke({"question": question})

logger.info("\n\n[DONE]", bright=True)