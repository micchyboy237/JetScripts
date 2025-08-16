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
# Composio

![Composio Example](img/ecosystem-composio.png)

Composio empowers AI agents to seamlessly connect with external tools, Apps, and APIs to perform actions and receive triggers. With built-in support for AutoGen, Composio enables the creation of highly capable and adaptable AI agents that can autonomously execute complex tasks and deliver personalized experiences.

- [Composio + AutoGen Documentation with Code Examples](https://docs.composio.dev/framework/autogen)
"""
logger.info("# Composio")

logger.info("\n\n[DONE]", bright=True)