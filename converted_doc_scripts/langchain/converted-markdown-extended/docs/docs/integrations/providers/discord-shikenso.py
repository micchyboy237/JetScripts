from jet.logger import logger
from langchain_discord.toolkits import DiscordToolkit
from langchain_discord.tools.discord_read_messages import DiscordReadMessages
from langchain_discord.tools.discord_send_messages import DiscordSendMessage
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
# Discord

> [Discord](https://discord.com/) is an instant messaging, voice, and video communication platform widely used by communities of all types.

## Installation and Setup

Install the `langchain-discord-shikenso` package:
"""
logger.info("# Discord")

pip install langchain-discord-shikenso

"""
You must provide a bot token via environment variable so the tools can authenticate with the Discord API:
"""
logger.info("You must provide a bot token via environment variable so the tools can authenticate with the Discord API:")

export DISCORD_BOT_TOKEN="your-discord-bot-token"

"""
If `DISCORD_BOT_TOKEN` is not set, the tools will raise a `ValueError` when instantiated.

---

## Tools

Below is a snippet showing how you can read and send messages in Discord. For more details, see the [documentation for Discord tools](/docs/integrations/tools/discord).
"""
logger.info("## Tools")


read_tool = DiscordReadMessages()
send_tool = DiscordSendMessage()

read_result = read_tool({"channel_id": "1234567890", "limit": 3})
logger.debug(read_result)

send_result = send_tool({"channel_id": "1234567890", "message": "Hello from Markdown example!"})
logger.debug(send_result)

"""
---

## Toolkit

`DiscordToolkit` groups multiple Discord-related tools into a single interface. For a usage example, see [the Discord toolkit docs](/docs/integrations/tools/discord).
"""
logger.info("## Toolkit")


toolkit = DiscordToolkit()
tools = toolkit.get_tools()

read_tool = tools[0]  # DiscordReadMessages
send_tool = tools[1]  # DiscordSendMessage

"""
---

## Future Integrations

Additional integrations (e.g., document loaders, chat loaders) could be added for Discord.
Check the [Discord Developer Docs](https://discord.com/developers/docs/intro) for more information, and watch for updates or advanced usage examples in the [langchain_discord GitHub repo](https://github.com/Shikenso-Analytics/langchain-discord).
"""
logger.info("## Future Integrations")

logger.info("\n\n[DONE]", bright=True)