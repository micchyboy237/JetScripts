

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Databricks

![Databricks Data Intelligence Platform](img/ecosystem-databricks.png)

The [Databricks Data Intelligence Platform ](https://www.databricks.com/product/data-intelligence-platform) allows your entire organization to use data and AI. It’s built on a lakehouse to provide an open, unified foundation for all data and governance, and is powered by a Data Intelligence Engine that understands the uniqueness of your data.


This example demonstrates how to use AutoGen alongside Databricks Foundation Model APIs and open-source LLM DBRX.

- [Databricks + AutoGen Code Examples](/docs/notebooks/agentchat_databricks_dbrx)
"""
logger.info("# Databricks")

logger.info("\n\n[DONE]", bright=True)