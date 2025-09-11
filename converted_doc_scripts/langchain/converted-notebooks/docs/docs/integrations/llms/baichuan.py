from jet.logger import logger
from langchain_community.llms import BaichuanLLM
import asyncio
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
# Baichuan LLM
Baichuan Inc. (https://www.baichuan-ai.com/) is a Chinese startup in the era of AGI, dedicated to addressing fundamental human needs: Efficiency, Health, and Happiness.
"""
logger.info("# Baichuan LLM")

# %pip install -qU langchain-community

"""
## Prerequisite
An API key is required to access Baichuan LLM API. Visit https://platform.baichuan-ai.com/ to get your API key.

## Use Baichuan LLM
"""
logger.info("## Prerequisite")


os.environ["BAICHUAN_API_KEY"] = "YOUR_API_KEY"


llm = BaichuanLLM()

res = llm.invoke("What's your name?")
logger.debug(res)

res = llm.generate(prompts=["你好！"])
res

for res in llm.stream("Who won the second world war?"):
    logger.debug(res)



async def run_aio_stream():
    for res in llm.stream("Write a poem about the sun."):
        logger.debug(res)


asyncio.run(run_aio_stream())

logger.info("\n\n[DONE]", bright=True)