from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.llms import CerebriumAI
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
# CerebriumAI

`Cerebrium` is an AWS Sagemaker alternative. It also provides API access to [several LLM models](https://docs.cerebrium.ai/cerebrium/prebuilt-models/deployment).

This notebook goes over how to use Langchain with [CerebriumAI](https://docs.cerebrium.ai/introduction).

## Install cerebrium
The `cerebrium` package is required to use the `CerebriumAI` API. Install `cerebrium` using `pip3 install cerebrium`.
"""
logger.info("# CerebriumAI")

# !pip3 install cerebrium

"""
## Imports
"""
logger.info("## Imports")



"""
## Set the Environment API Key
Make sure to get your API key from CerebriumAI. See [here](https://dashboard.cerebrium.ai/login). You are given a 1 hour free of serverless GPU compute to test different models.
"""
logger.info("## Set the Environment API Key")

os.environ["CEREBRIUMAI_API_KEY"] = "YOUR_KEY_HERE"

"""
## Create the CerebriumAI instance
You can specify different parameters such as the model endpoint url, max length, temperature, etc. You must provide an endpoint url.
"""
logger.info("## Create the CerebriumAI instance")

llm = CerebriumAI(endpoint_url="YOUR ENDPOINT URL HERE")

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