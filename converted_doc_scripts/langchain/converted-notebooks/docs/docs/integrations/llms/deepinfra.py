from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.llms import DeepInfra
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
# DeepInfra

[DeepInfra](https://deepinfra.com/?utm_source=langchain) is a serverless inference as a service that provides access to a [variety of LLMs](https://deepinfra.com/models?utm_source=langchain) and [embeddings models](https://deepinfra.com/models?type=embeddings&utm_source=langchain). This notebook goes over how to use LangChain with DeepInfra for language models.

## Set the Environment API Key
Make sure to get your API key from DeepInfra. You have to [Login](https://deepinfra.com/login?from=%2Fdash) and get a new token.

You are given a 1 hour free of serverless GPU compute to test different models. (see [here](https://github.com/deepinfra/deepctl#deepctl))
You can print your token with `deepctl auth token`
"""
logger.info("# DeepInfra")

# from getpass import getpass

# DEEPINFRA_API_TOKEN = getpass()


os.environ["DEEPINFRA_API_TOKEN"] = DEEPINFRA_API_TOKEN

"""
## Create the DeepInfra instance
You can also use our open-source [deepctl tool](https://github.com/deepinfra/deepctl#deepctl) to manage your model deployments. You can view a list of available parameters [here](https://deepinfra.com/databricks/dolly-v2-12b#API).
"""
logger.info("## Create the DeepInfra instance")


llm = DeepInfra(model_id="meta-llama/Llama-2-70b-chat-hf")
llm.model_kwargs = {
    "temperature": 0.7,
    "repetition_penalty": 1.2,
    "max_new_tokens": 250,
    "top_p": 0.9,
}

llm("Who let the dogs out?")

for chunk in llm.stream("Who let the dogs out?"):
    logger.debug(chunk)

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

question = "Can penguins reach the North pole?"

llm_chain.run(question)

logger.info("\n\n[DONE]", bright=True)