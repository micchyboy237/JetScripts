from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.llms import ForefrontAI
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
# ForefrontAI


The `Forefront` platform gives you the ability to fine-tune and use [open-source large language models](https://docs.forefront.ai/get-started/models).

This notebook goes over how to use Langchain with [ForefrontAI](https://www.forefront.ai/).

## Imports
"""
logger.info("# ForefrontAI")



"""
## Set the Environment API Key
Make sure to get your API key from ForefrontAI. You are given a 5 day free trial to test different models.
"""
logger.info("## Set the Environment API Key")

# from getpass import getpass

# FOREFRONTAI_API_KEY = getpass()

os.environ["FOREFRONTAI_API_KEY"] = FOREFRONTAI_API_KEY

"""
## Create the ForefrontAI instance
You can specify different parameters such as the model endpoint url, length, temperature, etc. You must provide an endpoint url.
"""
logger.info("## Create the ForefrontAI instance")

llm = ForefrontAI(endpoint_url="YOUR ENDPOINT URL HERE")

"""
## Create a Prompt Template
We will create a prompt template for Question and Answer.
"""
logger.info("## Create a Prompt Template")

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

"""
## Initiate the LLMChain
"""
logger.info("## Initiate the LLMChain")

llm_chain = LLMChain(prompt=prompt, llm=llm)

"""
## Run the LLMChain
Provide a question and run the LLMChain.
"""
logger.info("## Run the LLMChain")

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.run(question)

logger.info("\n\n[DONE]", bright=True)