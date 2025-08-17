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
title: '📽️ Youtube Channel'
---

## Setup

Make sure you have all the required packages installed before using this data type. You can install them by running the following command in your terminal.
"""
logger.info("## Setup")

pip install -U "embedchain[youtube]"

"""
## Usage

To add all the videos from a youtube channel to your app, use the data_type as `youtube_channel`.
"""
logger.info("## Usage")


app = App()
app.add("@channel_name", data_type="youtube_channel")

logger.info("\n\n[DONE]", bright=True)