from jet.logger import logger
from langchain_community.chat_models import VolcEngineMaasChat
from langchain_core.messages import HumanMessage
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
---
sidebar_label: Volc Engine Maas
---

# VolcEngineMaasChat

This notebook provides you with a guide on how to get started with volc engine maas chat models.
"""
logger.info("# VolcEngineMaasChat")

# %pip install --upgrade --quiet  volcengine


chat = VolcEngineMaasChat(volc_engine_maas_ak="your ak", volc_engine_maas_sk="your sk")

"""
or you can set access_key and secret_key in your environment variables
```bash
export VOLC_ACCESSKEY=YOUR_AK
export VOLC_SECRETKEY=YOUR_SK
```
"""
logger.info("or you can set access_key and secret_key in your environment variables")

chat([HumanMessage(content="给我讲个笑话")])

"""
# volc engine maas chat with stream
"""
logger.info("# volc engine maas chat with stream")

chat = VolcEngineMaasChat(
    volc_engine_maas_ak="your ak",
    volc_engine_maas_sk="your sk",
    streaming=True,
)

chat([HumanMessage(content="给我讲个笑话")])

logger.info("\n\n[DONE]", bright=True)