from jet.logger import logger
from langchain_community.document_loaders.discord import DiscordChatLoader
import os
import pandas as pd
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
# Discord

>[Discord](https://discord.com/) is a VoIP and instant messaging social platform. Users have the ability to communicate with voice calls, video calls, text messaging, media and files in private chats or as part of communities called "servers". A server is a collection of persistent chat rooms and voice channels which can be accessed via invite links.

Follow these steps to download your `Discord` data:

1. Go to your **User Settings**
2. Then go to **Privacy and Safety**
3. Head over to the **Request all of my Data** and click on **Request Data** button

It might take 30 days for you to receive your data. You'll receive an email at the address which is registered with Discord. That email will have a download button using which you would be able to download your personal Discord data.
"""
logger.info("# Discord")



path = input('Please enter the path to the contents of the Discord "messages" folder: ')
li = []
for f in os.listdir(path):
    expected_csv_path = os.path.join(path, f, "messages.csv")
    csv_exists = os.path.isfile(expected_csv_path)
    if csv_exists:
        df = pd.read_csv(expected_csv_path, index_col=None, header=0)
        li.append(df)

df = pd.concat(li, axis=0, ignore_index=True, sort=False)


loader = DiscordChatLoader(df, user_id_col="ID")
logger.debug(loader.load())

logger.info("\n\n[DONE]", bright=True)