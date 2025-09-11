from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI
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
# Cloudflare Workers AI

[Cloudflare AI documentation](https://developers.cloudflare.com/workers-ai/models/) listed all generative text models available.

Both Cloudflare account ID and API token are required. Find how to obtain them from [this document](https://developers.cloudflare.com/workers-ai/get-started/rest-api/).
"""
logger.info("# Cloudflare Workers AI")


template = """Human: {question}

AI Assistant: """

prompt = PromptTemplate.from_template(template)

"""
Get authentication before running LLM.
"""
logger.info("Get authentication before running LLM.")

# import getpass

# my_account_id = getpass.getpass("Enter your Cloudflare account ID:\n\n")
# my_api_token = getpass.getpass("Enter your Cloudflare API token:\n\n")
llm = CloudflareWorkersAI(account_id=my_account_id, api_token=my_api_token)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Why are roses red?"
llm_chain.run(question)

for chunk in llm.stream("Why is sky blue?"):
    logger.debug(chunk, end=" | ", flush=True)

logger.info("\n\n[DONE]", bright=True)