from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# component-schema-gen

This is a tool to generate schema for built in components.

Simply run `gen-component-schema` and it will print the schema to be used.
"""
logger.info("# component-schema-gen")

logger.info("\n\n[DONE]", bright=True)