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
title: '📄 Excel file'
---

### Excel file

To add any xlsx/xls file, use the data_type as `excel_file`. `excel_file` allows remote urls and conventional file paths. Eg:
"""
logger.info("### Excel file")


app = App()
app.add('https://example.com/content/intro.xlsx', data_type="excel_file")

app.query("Give brief information about data.")

logger.info("\n\n[DONE]", bright=True)