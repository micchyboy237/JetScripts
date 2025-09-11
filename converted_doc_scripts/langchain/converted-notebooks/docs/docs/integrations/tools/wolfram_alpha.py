from jet.logger import logger
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
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
# Wolfram Alpha

This notebook goes over how to use the wolfram alpha component.

First, you need to set up your Wolfram Alpha developer account and get your APP ID:

1. Go to wolfram alpha and sign up for a developer account [here](https://developer.wolframalpha.com/)
2. Create an app and get your APP ID
3. pip install wolframalpha

Then we will need to set some environment variables:
1. Save your APP ID into WOLFRAM_ALPHA_APPID env variable
"""
logger.info("# Wolfram Alpha")

pip install wolframalpha


os.environ["WOLFRAM_ALPHA_APPID"] = ""


wolfram = WolframAlphaAPIWrapper()

wolfram.run("What is 2x+5 = -3x + 7?")

logger.info("\n\n[DONE]", bright=True)