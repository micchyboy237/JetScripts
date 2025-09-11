from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain.chains import LLMMathChain
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
# Math chain

This notebook showcases using LLMs and Python REPLs to do complex word math problems.
"""
logger.info("# Math chain")


llm = Ollama(temperature=0)
llm_math = LLMMathChain.from_llm(llm, verbose=True)

llm_math.invoke("What is 13 raised to the .3432 power?")

logger.info("\n\n[DONE]", bright=True)