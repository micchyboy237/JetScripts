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
title: '🧾 XML file'
---

### XML file

To add any xml file, use the data_type as `xml`. Eg:
"""
logger.info("### XML file")


app = App()

app.add('content/data.xml')

"""
Note: Only the text content of the xml file will be added to the app. The tags will be ignored.
"""
logger.info("Note: Only the text content of the xml file will be added to the app. The tags will be ignored.")

logger.info("\n\n[DONE]", bright=True)