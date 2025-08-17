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
title: "🐝 Beehiiv"
---

To add any Beehiiv data sources to your app, just add the base url as the source and set the data_type to `beehiiv`.
"""
logger.info("title: "🐝 Beehiiv"")


app = App()

app.add('https://aibreakfast.beehiiv.com', data_type='beehiiv')
app.query("How much is MLX paying developers?")

logger.info("\n\n[DONE]", bright=True)