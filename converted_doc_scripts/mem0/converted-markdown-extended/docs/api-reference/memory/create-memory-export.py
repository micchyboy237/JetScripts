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
---
title: 'Create Memory Export'
openapi: post /v1/exports/
---

Submit a job to create a structured export of memories using a customizable Pydantic schema. This process may take some time to complete, especially if you’re exporting a large number of memories. You can tailor the export by applying various filters (e.g., user_id, agent_id, run_id, or session_id) and by modifying the Pydantic schema to ensure the final data matches your exact needs.
"""
logger.info("title: 'Create Memory Export'")

logger.info("\n\n[DONE]", bright=True)