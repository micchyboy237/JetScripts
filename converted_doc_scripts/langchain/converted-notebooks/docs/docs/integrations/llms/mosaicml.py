from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.llms import MosaicML
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
# MosaicML

[MosaicML](https://docs.mosaicml.com/en/latest/inference.html) offers a managed inference service. You can either use a variety of open-source models, or deploy your own.

This example goes over how to use LangChain to interact with MosaicML Inference for text completion.
"""
logger.info("# MosaicML")

# from getpass import getpass

# MOSAICML_API_TOKEN = getpass()


os.environ["MOSAICML_API_TOKEN"] = MOSAICML_API_TOKEN


template = """Question: {question}"""

prompt = PromptTemplate.from_template(template)

llm = MosaicML(inject_instruction_format=True, model_kwargs={"max_new_tokens": 128})

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What is one good reason why you should train a large language model on domain specific data?"

llm_chain.run(question)

logger.info("\n\n[DONE]", bright=True)