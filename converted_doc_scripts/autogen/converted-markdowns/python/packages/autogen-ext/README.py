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
# AutoGen Extensions

- [Documentation](https://microsoft.github.io/autogen/stable/user-guide/extensions-user-guide/index.html)

AutoGen is designed to be extensible. The `autogen-ext` package contains many different component implementations maintained by the AutoGen project. However, we strongly encourage others to build their own components and publish them as part of the ecosytem.
"""
logger.info("# AutoGen Extensions")

logger.info("\n\n[DONE]", bright=True)