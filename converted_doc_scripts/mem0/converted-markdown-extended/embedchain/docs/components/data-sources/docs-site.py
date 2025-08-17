from embedchain import App
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
title: '📚 Code Docs website'
---

To add any code documentation website as a loader, use the data_type as `docs_site`. Eg:
"""
logger.info("title: '📚 Code Docs website'")


app = App()
app.add("https://docs.embedchain.ai/", data_type="docs_site")
app.query("What is Embedchain?")

logger.info("\n\n[DONE]", bright=True)