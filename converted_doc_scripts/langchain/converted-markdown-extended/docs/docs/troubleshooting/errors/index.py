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
# Error reference

This page contains guides around resolving common errors you may find while building with LangChain.
Errors referenced below will have an `lc_error_code` property corresponding to one of the below codes when they are thrown in code.

- [INVALID_PROMPT_INPUT](/docs/troubleshooting/errors/INVALID_PROMPT_INPUT)
- [INVALID_TOOL_RESULTS](/docs/troubleshooting/errors/INVALID_TOOL_RESULTS)
- [MESSAGE_COERCION_FAILURE](/docs/troubleshooting/errors/MESSAGE_COERCION_FAILURE)
- [MODEL_AUTHENTICATION](/docs/troubleshooting/errors/MODEL_AUTHENTICATION)
- [MODEL_NOT_FOUND](/docs/troubleshooting/errors/MODEL_NOT_FOUND)
- [MODEL_RATE_LIMIT](/docs/troubleshooting/errors/MODEL_RATE_LIMIT)
- [OUTPUT_PARSING_FAILURE](/docs/troubleshooting/errors/OUTPUT_PARSING_FAILURE)
"""
logger.info("# Error reference")

logger.info("\n\n[DONE]", bright=True)