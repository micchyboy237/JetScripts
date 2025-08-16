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
# **Multi-Agent PostgreSQL Data Management System with AutoGen and Azure PostgreSQL**


<div align="center">
  <img src="https://github.com/mehrsa/MultiAgent_Azure_PostgreSQL_AutoGen0.4/blob/main/misc/Drawing%203.png" alt="Architecture">
</div>

Go to below repository to try out a demo demonstrating how to build a **multi-agent AI system** for managing shipment data stored on an Azure PostgreSQL database:

[MultiAgent_Azure_PostgreSQL_AutoGen](https://github.com/Azure-Samples/MultiAgent_Azure_PostgreSQL_AutoGen0.4/tree/main)
"""
logger.info("# **Multi-Agent PostgreSQL Data Management System with AutoGen and Azure PostgreSQL**")

logger.info("\n\n[DONE]", bright=True)