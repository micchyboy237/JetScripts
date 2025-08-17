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
title: 'Get Memory Export'
openapi: post /v1/exports/get
---

Retrieve the latest structured memory export after submitting an export job. You can filter the export by `user_id`, `run_id`, `session_id`, or `app_id` to get the most recent export matching your filters.
"""
logger.info("title: 'Get Memory Export'")

logger.info("\n\n[DONE]", bright=True)