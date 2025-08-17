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
title: 'Delete Webhook'
openapi: delete /api/v1/webhooks/{webhook_id}/
---

## Delete Webhook

Delete a webhook by providing the webhook ID.
"""
logger.info("## Delete Webhook")

logger.info("\n\n[DONE]", bright=True)