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
title: '📄 Text file'
---

To add a .txt file, specify the data_type as `text_file`. The URL provided in the first parameter of the `add` function, should be a local path. Eg:
"""
logger.info("title: '📄 Text file'")


app = App()
app.add('path/to/file.txt', data_type="text_file")

app.query("Summarize the information of the text file")

logger.info("\n\n[DONE]", bright=True)