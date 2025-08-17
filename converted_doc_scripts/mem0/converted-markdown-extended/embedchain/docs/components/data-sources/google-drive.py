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
title: 'Google Drive'
---

To use GoogleDriveLoader you must install the extra dependencies with `pip install --upgrade embedchain[googledrive]`.

The data_type must be `google_drive`. Otherwise, it will be considered a regular web page.

Google Drive requires the setup of credentials. This can be done by following the steps below:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/apis/credentials).
2. Create a project if you don't have one already.
3. Enable the [Google Drive API](https://console.cloud.google.com/flows/enableapi?apiid=drive.googleapis.com)
4. [Authorize credentials for desktop app](https://developers.google.com/drive/api/quickstart/python#authorize_credentials_for_a_desktop_application)
5. When done, you will be able to download the credentials in `json` format. Rename the downloaded file to `credentials.json` and save it in `~/.credentials/credentials.json`
6. Set the environment variable `GOOGLE_APPLICATION_CREDENTIALS=~/.credentials/credentials.json`

The first time you use the loader, you will be prompted to enter your Google account credentials.
"""
logger.info("title: 'Google Drive'")


app = App()

url = "https://drive.google.com/drive/u/0/folders/xxx-xxx"
app.add(url, data_type="google_drive")

logger.info("\n\n[DONE]", bright=True)