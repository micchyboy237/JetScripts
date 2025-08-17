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
title: '💾 Dropbox'
---

To load folders or files from your Dropbox account, configure the `data_type` parameter as `dropbox` and specify the path to the desired file or folder, starting from the root directory of your Dropbox account.

For Dropbox access, an **access token** is required. Obtain this token by visiting [Dropbox Developer Apps](https://www.dropbox.com/developers/apps). There, create a new app and generate an access token for it.

Ensure your app has the following settings activated:

- In the Permissions section, enable `files.content.read` and `files.metadata.read`.

## Usage

Install the `dropbox` pypi package:
"""
logger.info("## Usage")

pip install dropbox

"""
Following is an example of how to use the dropbox loader:
"""
logger.info("Following is an example of how to use the dropbox loader:")


os.environ["DROPBOX_ACCESS_TOKEN"] = "sl.xxx"
# os.environ["OPENAI_API_KEY"] = "sk-xxx"

app = App()

app.add("/test", data_type="dropbox")

logger.debug(app.query("Which two celebrities are mentioned here?"))

logger.info("\n\n[DONE]", bright=True)