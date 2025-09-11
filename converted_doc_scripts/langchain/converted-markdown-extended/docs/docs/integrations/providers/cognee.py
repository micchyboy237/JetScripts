from jet.logger import logger
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
# Cognee

Cognee implements scalable, modular ECL (Extract, Cognify, Load) pipelines that allow
you to interconnect and retrieve past conversations, documents, and audio
transcriptions while reducing hallucinations, developer effort, and cost.

Cognee merges graph and vector databases to uncover hidden relationships and new
patterns in your data. You can automatically model, load and retrieve entities and
objects representing your business domain and analyze their relationships, uncovering
insights that neither vector stores nor graph stores alone can provide.

Try it in a Google Colab  <a href="https://colab.research.google.com/drive/1g-Qnx6l_ecHZi0IOw23rg0qC4TYvEvWZ?usp=sharing">notebook</a>  or have a look at the <a href="https://docs.cognee.ai">documentation</a>.

If you have questions, join cognee <a href="https://discord.gg/NQPKmU5CCg">Discord</a> community.

Have you seen cognee's <a href="https://github.com/topoteretes/cognee-starter">starter repo</a>? Check it out!


## Installation and Setup
"""
logger.info("# Cognee")

pip install langchain-cognee

"""
## Retrievers

See detail on available retrievers [here](/docs/integrations/retrievers/cognee).
"""
logger.info("## Retrievers")

logger.info("\n\n[DONE]", bright=True)