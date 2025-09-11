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
# How-to guides

- [**Documentation**](documentation/index.mdx): Help improve our docs, including this one!
- [**Code**](code/index.mdx): Help us write code, fix bugs, or improve our infrastructure.

## Integrations

- [**Start Here**](integrations/index.mdx): Help us integrate with your favorite vendors and tools.
- [**Package**](integrations/package): Publish an integration package to PyPi
- [**Standard Tests**](integrations/standard_tests): Ensure your integration passes an expected set of tests.
"""
logger.info("# How-to guides")

logger.info("\n\n[DONE]", bright=True)