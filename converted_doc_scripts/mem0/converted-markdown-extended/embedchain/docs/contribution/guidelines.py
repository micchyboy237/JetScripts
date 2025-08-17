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
title: '📋 Guidelines'
url: https://github.com/mem0ai/mem0/blob/main/embedchain/CONTRIBUTING.md
---
"""
logger.info("title: '📋 Guidelines'")

logger.info("\n\n[DONE]", bright=True)