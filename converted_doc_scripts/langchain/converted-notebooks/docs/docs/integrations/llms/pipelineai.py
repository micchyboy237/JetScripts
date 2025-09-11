from jet.logger import logger
from langchain_community.llms import PipelineAI
from langchain_core.output_parsers import StrOutputParser
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
# PipelineAI

>[PipelineAI](https://pipeline.ai) allows you to run your ML models at scale in the cloud. It also provides API access to [several LLM models](https://pipeline.ai).

This notebook goes over how to use Langchain with [PipelineAI](https://docs.pipeline.ai/docs).

## PipelineAI example

[This example shows how PipelineAI integrated with LangChain](https://docs.pipeline.ai/docs/langchain) and it is created by PipelineAI.

## Setup
The `pipeline-ai` library is required to use the `PipelineAI` API, AKA `Pipeline Cloud`. Install `pipeline-ai` using `pip install pipeline-ai`.
"""
logger.info("# PipelineAI")

# %pip install --upgrade --quiet  pipeline-ai

"""
## Example

### Imports
"""
logger.info("## Example")



"""
### Set the Environment API Key
Make sure to get your API key from PipelineAI. Check out the [cloud quickstart guide](https://docs.pipeline.ai/docs/cloud-quickstart). You'll be given a 30 day free trial with 10 hours of serverless GPU compute to test different models.
"""
logger.info("### Set the Environment API Key")

os.environ["PIPELINE_API_KEY"] = "YOUR_API_KEY_HERE"

"""
## Create the PipelineAI instance
When instantiating PipelineAI, you need to specify the id or tag of the pipeline you want to use, e.g. `pipeline_key = "public/gpt-j:base"`. You then have the option of passing additional pipeline-specific keyword arguments:
"""
logger.info("## Create the PipelineAI instance")

llm = PipelineAI(pipeline_key="YOUR_PIPELINE_KEY", pipeline_kwargs={...})

"""
### Create a Prompt Template
We will create a prompt template for Question and Answer.
"""
logger.info("### Create a Prompt Template")

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

"""
### Initiate the LLMChain
"""
logger.info("### Initiate the LLMChain")

llm_chain = prompt | llm | StrOutputParser()

"""
### Run the LLMChain
Provide a question and run the LLMChain.
"""
logger.info("### Run the LLMChain")

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.invoke(question)

logger.info("\n\n[DONE]", bright=True)