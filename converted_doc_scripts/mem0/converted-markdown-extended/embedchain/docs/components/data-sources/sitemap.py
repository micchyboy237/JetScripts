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
title: '🗺️ Sitemap'
---

Add all web pages from an xml-sitemap. Filters non-text files. Use the data_type as `sitemap`. Eg:
"""
logger.info("title: '🗺️ Sitemap'")


app = App()

app.add('https://example.com/sitemap.xml', data_type='sitemap')

logger.info("\n\n[DONE]", bright=True)