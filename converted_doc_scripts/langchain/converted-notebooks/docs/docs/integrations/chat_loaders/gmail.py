from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from jet.logger import logger
from langchain_community.chat_loaders.gmail import GMailLoader
from langchain_community.chat_loaders.utils import (
map_ai_messages,
)
import os
import os.path
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# GMail

This loader goes over how to load data from GMail. There are many ways you could want to load data from GMail. This loader is currently fairly opinionated in how to do so. The way it does it is it first looks for all messages that you have sent. It then looks for messages where you are responding to a previous email. It then fetches that previous email, and creates a training example of that email, followed by your email.

Note that there are clear limitations here. For example, all examples created are only looking at the previous email for context.

To use:

- Set up a Google Developer Account: Go to the Google Developer Console, create a project, and enable the Gmail API for that project. This will give you a credentials.json file that you'll need later.

- Install the Google Client Library: Run the following command to install the Google Client Library:
"""
logger.info("# GMail")

# %pip install --upgrade --quiet  google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client



SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


creds = None
if os.path.exists("email_token.json"):
    creds = Credentials.from_authorized_user_file("email_token.json", SCOPES)
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            "creds.json",
            SCOPES,
        )
        creds = flow.run_local_server(port=0)
    with open("email_token.json", "w") as token:
        token.write(creds.to_json())


loader = GMailLoader(creds=creds, n=3)

data = loader.load()

len(data)


training_data = list(
    map_ai_messages(data, sender="Harrison Chase <hchase@langchain.com>")
)

logger.info("\n\n[DONE]", bright=True)