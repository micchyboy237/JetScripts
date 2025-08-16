from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# MemGPT

![MemGPT Example](img/ecosystem-memgpt.png)

MemGPT enables LLMs to manage their own memory and overcome limited context windows. You can use MemGPT to create perpetual chatbots that learn about you and modify their own personalities over time. You can connect MemGPT to your own local filesystems and databases, as well as connect MemGPT to your own tools and APIs. The MemGPT + AutoGen integration allows you to equip any AutoGen agent with MemGPT capabilities.

- [MemGPT + AutoGen Documentation with Code Examples](https://memgpt.readme.io/docs/autogen)
"""
logger.info("# MemGPT")

logger.info("\n\n[DONE]", bright=True)