from jet.logger import logger
from langchain_community.chat_models import ChatYuan2
from langchain_community.llms.yuan2 import Yuan2
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
# IEIT Systems

>[IEIT Systems](https://en.ieisystem.com/) is a Chinese information technology company
> established in 1999. It provides the IT infrastructure products, solutions,
> and services, innovative IT products and solutions across cloud computing,
> big data, and artificial intelligence.


## LLMs

See a [usage example](/docs/integrations/llms/yuan2).
"""
logger.info("# IEIT Systems")


"""
## Chat models

See the [installation instructions](/docs/integrations/chat/yuan2/#setting-up-your-api-server).

Yuan2.0 provided an Ollama compatible API, and ChatYuan2 is integrated into langchain by using `Ollama client`.
Therefore, ensure the `ollama` package is installed.
"""
logger.info("## Chat models")

pip install ollama

"""
See a [usage example](/docs/integrations/chat/yuan2).
"""
logger.info("See a [usage example](/docs/integrations/chat/yuan2).")


logger.info("\n\n[DONE]", bright=True)