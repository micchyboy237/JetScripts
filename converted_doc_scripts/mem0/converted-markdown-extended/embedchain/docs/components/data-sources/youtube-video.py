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
title: '📺 Youtube Video'
---

## Setup

Make sure you have all the required packages installed before using this data type. You can install them by running the following command in your terminal.
"""
logger.info("## Setup")

pip install -U "embedchain[youtube]"

"""
## Usage

To add any youtube video to your app, use the data_type as `youtube_video`. Eg:
"""
logger.info("## Usage")


app = App()
app.add('a_valid_youtube_url_here', data_type='youtube_video')

logger.info("\n\n[DONE]", bright=True)