from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.llms import OpenLM
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
# OpenLM
[OpenLM](https://github.com/r2d4/openlm) is a zero-dependency Ollama-compatible LLM provider that can call different inference endpoints directly via HTTP. 


It implements the Ollama Completion class so that it can be used as a drop-in replacement for the Ollama API. This changeset utilizes BaseOllama for minimal added code.

This examples goes over how to use LangChain to interact with both Ollama and HuggingFace. You'll need API keys from both.

### Setup
Install dependencies and set API keys.
"""
logger.info("# OpenLM")

# %pip install --upgrade --quiet  openlm
# %pip install --upgrade --quiet  langchain-ollama

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
    logger.debug("Enter your Ollama API key:")
#     os.environ["OPENAI_API_KEY"] = getpass()

if "HF_API_TOKEN" not in os.environ:
    logger.debug("Enter your HuggingFace Hub API key:")
#     os.environ["HF_API_TOKEN"] = getpass()

"""
### Using LangChain with OpenLM

Here we're going to call two models in an LLMChain, `text-davinci-003` from Ollama and `gpt2` on HuggingFace.
"""
logger.info("### Using LangChain with OpenLM")


question = "What is the capital of France?"
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

for model in ["text-davinci-003", "huggingface.co/gpt2"]:
    llm = OpenLM(model=model)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    result = llm_chain.run(question)
    logger.debug(
        """Model: {}
Result: {}""".format(model, result)
    )

logger.info("\n\n[DONE]", bright=True)