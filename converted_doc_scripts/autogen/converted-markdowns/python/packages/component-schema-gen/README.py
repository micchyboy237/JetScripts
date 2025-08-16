

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# component-schema-gen

This is a tool to generate schema for built in components.

Simply run `gen-component-schema` and it will print the schema to be used.
"""
logger.info("# component-schema-gen")

logger.info("\n\n[DONE]", bright=True)