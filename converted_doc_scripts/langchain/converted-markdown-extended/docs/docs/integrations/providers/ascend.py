from jet.logger import logger
from langchain_community.embeddings import AscendEmbeddings
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
# Ascend

>[Ascend](https://https://www.hiascend.com/) is Natural Process Unit provide by Huawei

This page covers how to use ascend NPU with LangChain.

### Installation

Install using torch-npu using:
"""
logger.info("# Ascend")

pip install torch-npu

"""
Please follow the installation instructions as specified below:
* Install CANN as shown [here](https://www.hiascend.com/document/detail/zh/canncommercial/700/quickstart/quickstart/quickstart_18_0002.html).

### Embedding Models

See a [usage example](/docs/integrations/text_embedding/ascend).
"""
logger.info("### Embedding Models")


logger.info("\n\n[DONE]", bright=True)