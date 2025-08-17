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
title: 🔄 reset
---

`reset()` method allows you to wipe the data from your RAG application and start from scratch.

## Usage
"""
logger.info("## Usage")


app = App()
app.add("https://www.forbes.com/profile/elon-musk")

app.reset()

logger.info("\n\n[DONE]", bright=True)