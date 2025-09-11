from jet.logger import logger
from langchain_community.agent_toolkits.load_tools import load_huggingface_tool
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
# HuggingFace Hub Tools

>[Huggingface Tools](https://huggingface.co/docs/transformers/v4.29.0/en/custom_tools) that supporting text I/O can be
loaded directly using the `load_huggingface_tool` function.
"""
logger.info("# HuggingFace Hub Tools")

# %pip install --upgrade --quiet  transformers huggingface_hub > /dev/null

# %pip install --upgrade --quiet  langchain-community


tool = load_huggingface_tool("lysandre/hf-model-downloads")

logger.debug(f"{tool.name}: {tool.description}")

tool.run("text-classification")

logger.info("\n\n[DONE]", bright=True)