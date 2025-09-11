from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.llms import GooseAI
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
# GooseAI

`GooseAI` is a fully managed NLP-as-a-Service, delivered via API. GooseAI provides access to [these models](https://goose.ai/docs/models).

This notebook goes over how to use Langchain with [GooseAI](https://goose.ai/).

## Install ollama
The `ollama` package is required to use the GooseAI API. Install `ollama` using `pip install ollama`.
"""
logger.info("# GooseAI")

# %pip install --upgrade --quiet  langchain-ollama

"""
## Imports
"""
logger.info("## Imports")



"""
## Set the Environment API Key
Make sure to get your API key from GooseAI. You are given $10 in free credits to test different models.
"""
logger.info("## Set the Environment API Key")

# from getpass import getpass

# GOOSEAI_API_KEY = getpass()

os.environ["GOOSEAI_API_KEY"] = GOOSEAI_API_KEY

"""
## Create the GooseAI instance
You can specify different parameters such as the model name, max tokens generated, temperature, etc.
"""
logger.info("## Create the GooseAI instance")

llm = GooseAI()

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