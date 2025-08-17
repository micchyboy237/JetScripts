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
title: '📓 Notion'
---

To use notion you must install the extra dependencies with `pip install --upgrade embedchain[community]`.

To load a notion page, use the data_type as `notion`. Since it is hard to automatically detect, it is advised to specify the `data_type` when adding a notion document.
The next argument must **end** with the `notion page id`. The id is a 32-character string. Eg:
"""
logger.info("title: '📓 Notion'")


app = App()

app.add("cfbc134ca6464fc980d0391613959196", data_type="notion")
app.add("my-page-cfbc134ca6464fc980d0391613959196", data_type="notion")
app.add("https://www.notion.so/my-page-cfbc134ca6464fc980d0391613959196", data_type="notion")

app.query("Summarize the notion doc")

logger.info("\n\n[DONE]", bright=True)