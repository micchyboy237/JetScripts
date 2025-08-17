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
title: '📝 Mdx file'
---

To add any `.mdx` file to your app, use the data_type (first argument to `.add()` method) as `mdx`. Note that this supports support mdx file present on machine, so this should be a file path. Eg:
"""
logger.info("title: '📝 Mdx file'")


app = App()
app.add('path/to/file.mdx', data_type='mdx')

app.query("What are the docs about?")

logger.info("\n\n[DONE]", bright=True)