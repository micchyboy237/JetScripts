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
title: '📄 Docx file'
---

### Docx file

To add any doc/docx file, use the data_type as `docx`. `docx` allows remote urls and conventional file paths. Eg:
"""
logger.info("### Docx file")


app = App()
app.add('https://example.com/content/intro.docx', data_type="docx")

app.query("Summarize the docx data?")

logger.info("\n\n[DONE]", bright=True)