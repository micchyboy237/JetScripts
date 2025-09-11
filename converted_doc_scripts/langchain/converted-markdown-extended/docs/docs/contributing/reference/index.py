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
# Reference

- [**Repository Structure**](repo_structure.mdx): Understand the high level structure of the repository.
- [**Review Process**](review_process.mdx): Learn about the review process for pull requests.
- [**Frequently Asked Questions (FAQ)**](faq.mdx): Get answers to common questions about contributing.
"""
logger.info("# Reference")

logger.info("\n\n[DONE]", bright=True)