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
title: '🌐 HTML Web page'
---

To add any web page, use the data_type as `web_page`. Eg:
"""
logger.info("title: '🌐 HTML Web page'")


app = App()

app.add('a_valid_web_page_url', data_type='web_page')

logger.info("\n\n[DONE]", bright=True)