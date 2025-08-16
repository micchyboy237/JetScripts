from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# AutoGen Extensions

- [Documentation](https://microsoft.github.io/autogen/stable/user-guide/extensions-user-guide/index.html)

AutoGen is designed to be extensible. The `autogen-ext` package contains many different component implementations maintained by the AutoGen project. However, we strongly encourage others to build their own components and publish them as part of the ecosytem.
"""
logger.info("# AutoGen Extensions")

logger.info("\n\n[DONE]", bright=True)