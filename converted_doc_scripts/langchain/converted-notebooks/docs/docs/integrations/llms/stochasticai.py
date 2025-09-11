from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.llms import StochasticAI
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
# StochasticAI

>[Stochastic Acceleration Platform](https://docs.stochastic.ai/docs/introduction/) aims to simplify the life cycle of a Deep Learning model. From uploading and versioning the model, through training, compression and acceleration to putting it into production.

This example goes over how to use LangChain to interact with `StochasticAI` models.

You have to get the API_KEY and the API_URL [here](https://app.stochastic.ai/workspace/profile/settings?tab=profile).
"""
logger.info("# StochasticAI")

# from getpass import getpass

# STOCHASTICAI_API_KEY = getpass()


os.environ["STOCHASTICAI_API_KEY"] = STOCHASTICAI_API_KEY

# YOUR_API_URL = getpass()


template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

llm = StochasticAI(api_url=YOUR_API_URL)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.run(question)

logger.info("\n\n[DONE]", bright=True)