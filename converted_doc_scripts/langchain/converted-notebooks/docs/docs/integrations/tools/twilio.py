from jet.logger import logger
from langchain_community.utilities.twilio import TwilioAPIWrapper
import os
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
# Twilio

This notebook goes over how to use the [Twilio](https://www.twilio.com) API wrapper to send a message through SMS or [Twilio Messaging Channels](https://www.twilio.com/docs/messaging/channels).

Twilio Messaging Channels facilitates integrations with 3rd party messaging apps and lets you send messages through WhatsApp Business Platform (GA), Facebook Messenger (Public Beta) and Google Business Messages (Private Beta).

## Setup

To use this tool you need to install the Python Twilio package `twilio`
"""
logger.info("# Twilio")

# %pip install --upgrade --quiet  twilio

"""
You'll also need to set up a Twilio account and get your credentials. You'll need your Account String Identifier (SID) and your Auth Token. You'll also need a number to send messages from.

You can either pass these in to the TwilioAPIWrapper as named parameters `account_sid`, `auth_token`, `from_number`, or you can set the environment variables `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`.

## Sending an SMS
"""
logger.info("## Sending an SMS")


twilio = TwilioAPIWrapper(
)

twilio.run("hello world", "+16162904619")

"""
## Sending a WhatsApp Message

You'll need to link your WhatsApp Business Account with Twilio. You'll also need to make sure that the number to send messages from is configured as a WhatsApp Enabled Sender on Twilio and registered with WhatsApp.
"""
logger.info("## Sending a WhatsApp Message")


twilio = TwilioAPIWrapper(
)

twilio.run("hello world", "whatsapp: +16162904619")

logger.info("\n\n[DONE]", bright=True)