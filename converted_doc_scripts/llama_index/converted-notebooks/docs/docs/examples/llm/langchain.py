from jet.logger import CustomLogger
from langchain.llms import MLX
from llama_index.llms.langchain import LangChainLLM
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/langchain.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# LangChain LLM
"""
logger.info("# LangChain LLM")

# %pip install llama-index-llms-langchain



llm = LangChainLLM(llm=MLX())

response_gen = llm.stream_complete("Hi this is")

for delta in response_gen:
    logger.debug(delta.delta, end="")

logger.info("\n\n[DONE]", bright=True)