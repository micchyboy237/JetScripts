from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.llms import Petals
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
# Petals

`Petals` runs 100B+ language models at home, BitTorrent-style.

This notebook goes over how to use Langchain with [Petals](https://github.com/bigscience-workshop/petals).

## Install petals
The `petals` package is required to use the Petals API. Install `petals` using `pip3 install petals`.

For Apple Silicon(M1/M2) users please follow this guide [https://github.com/bigscience-workshop/petals/issues/147#issuecomment-1365379642](https://github.com/bigscience-workshop/petals/issues/147#issuecomment-1365379642) to install petals
"""
logger.info("# Petals")

# !pip3 install petals

"""
## Imports
"""
logger.info("## Imports")



"""
## Set the Environment API Key
Make sure to get [your API key](https://huggingface.co/docs/api-inference/quicktour#get-your-api-token) from Huggingface.
"""
logger.info("## Set the Environment API Key")

# from getpass import getpass

# HUGGINGFACE_API_KEY = getpass()

os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY

"""
## Create the Petals instance
You can specify different parameters such as the model name, max new tokens, temperature, etc.
"""
logger.info("## Create the Petals instance")

llm = Petals(model_name="bigscience/bloom-petals")

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