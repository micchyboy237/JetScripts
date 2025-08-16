from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# AutoGen Core

- [Documentation](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/index.html)

AutoGen core offers an easy way to quickly build event-driven, distributed, scalable, resilient AI agent systems. Agents are developed by using the [Actor model](https://en.wikipedia.org/wiki/Actor_model). You can build and run your agent system locally and easily move to a distributed system in the cloud when you are ready.
"""
logger.info("# AutoGen Core")

logger.info("\n\n[DONE]", bright=True)