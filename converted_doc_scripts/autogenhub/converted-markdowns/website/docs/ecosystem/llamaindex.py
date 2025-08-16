from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Llamaindex

![Llamaindex Example](img/ecosystem-llamaindex.png)

[Llamaindex](https://www.llamaindex.ai/) allows the users to create Llamaindex agents and integrate them in autogen conversation patterns.

- [Llamaindex + AutoGen Code Examples](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_group_chat_with_llamaindex_agents.ipynb)
"""
logger.info("# Llamaindex")

logger.info("\n\n[DONE]", bright=True)