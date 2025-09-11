from jet.logger import logger
from langchain_community.llms.moonshot import Moonshot
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
# MoonshotChat

[Moonshot](https://platform.moonshot.cn/) is a Chinese startup that provides LLM service for companies and individuals.

This example goes over how to use LangChain to interact with Moonshot.
"""
logger.info("# MoonshotChat")



os.environ["MOONSHOT_API_KEY"] = "MOONSHOT_API_KEY"

llm = Moonshot()

llm.invoke("What is the difference between panda and bear?")

logger.info("\n\n[DONE]", bright=True)