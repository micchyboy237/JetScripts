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
title: "💬 Discord"
---

To add any Discord channel messages to your app, just add the `channel_id` as the source and set the `data_type` to `discord`.

<Note>
    This loader requires a Discord bot token with read messages access.
    To obtain the token, follow the instructions provided in this tutorial: 
    <a href="https://www.writebots.com/discord-bot-token/">How to Get a Discord Bot Token?</a>.
</Note>
"""
logger.info("title: "💬 Discord"")


os.environ["DISCORD_TOKEN"] = "xxx"

app = App()

app.add("1177296711023075338", data_type="discord")

response = app.query("What is Joe saying about Elon Musk?")

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)