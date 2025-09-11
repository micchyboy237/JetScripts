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
# Contribute documentation

Documentation is a vital part of LangChain. We welcome both new documentation for new features and
community improvements to our current documentation. Please read the resources below before getting started:

- [Documentation style guide](style_guide.mdx)
- [Setup](setup.mdx)
"""
logger.info("# Contribute documentation")

logger.info("\n\n[DONE]", bright=True)