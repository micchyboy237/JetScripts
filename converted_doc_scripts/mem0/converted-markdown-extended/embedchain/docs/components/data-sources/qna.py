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
title: '❓💬 Question and answer pair'
---

QnA pair is a local data type. To supply your own QnA pair, use the data_type as `qna_pair` and enter a tuple. Eg:
"""
logger.info("title: '❓💬 Question and answer pair'")


app = App()

app.add(("Question", "Answer"), data_type="qna_pair")

logger.info("\n\n[DONE]", bright=True)