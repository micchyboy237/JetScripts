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
title: 🚀 deploy
---

The `deploy()` method is currently available on an invitation-only basis. To request access, please submit your information via the provided [Google Form](https://forms.gle/vigN11h7b4Ywat668). We will review your request and respond promptly.
"""
logger.info("title: 🚀 deploy")

logger.info("\n\n[DONE]", bright=True)