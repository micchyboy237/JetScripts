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
title: 'Add Member'
openapi: post /api/v1/orgs/organizations/{org_id}/projects/{project_id}/members/
---

The API provides two roles for project members:

- `READER`: Allows viewing of project resources.
- `OWNER`: Grants full administrative access to manage the project and its resources.
"""
logger.info("title: 'Add Member'")

logger.info("\n\n[DONE]", bright=True)