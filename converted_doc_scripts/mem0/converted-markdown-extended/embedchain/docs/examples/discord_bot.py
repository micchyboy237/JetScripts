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
title: "🤖 Discord Bot"
---

### 🔑 Keys Setup

# - Set your `OPENAI_API_KEY` in your variables.env file.
- Go to [https://discord.com/developers/applications/](https://discord.com/developers/applications/) and click on `New Application`.
- Enter the name for your bot, accept the terms and click on `Create`. On the resulting page, enter the details of your bot as you like.
- On the left sidebar, click on `Bot`. Under the heading `Privileged Gateway Intents`, toggle all 3 options to ON position. Save your changes.
- Now click on `Reset Token` and copy the token value. Set it as `DISCORD_BOT_TOKEN` in .env file.
- On the left sidebar, click on `OAuth2` and go to `General`.
- Set `Authorization Method` to `In-app Authorization`. Under `Scopes` select `bot`.
- Under `Bot Permissions` allow the following and then click on `Save Changes`.

Send Messages (under Text Permissions)

- Now under `OAuth2` and go to `URL Generator`. Under `Scopes` select `bot`.
- Under `Bot Permissions` set the same permissions as above.
- Now scroll down and copy the `Generated URL`. Paste it in a browser window and select the Server where you want to add the bot.
- Click on `Continue` and authorize the bot.
- 🎉 The bot has been successfully added to your server. But it's still offline.

### Take the bot online

<Tabs>
    <Tab title="docker">
        ```bash
#         docker run --name discord-bot -e OPENAI_API_KEY=sk-xxx -e DISCORD_BOT_TOKEN=xxx -p 8080:8080 embedchain/discord-bot:latest
        ```
    </Tab>
    <Tab title="python">
        ```bash
#         pip install --upgrade "embedchain[discord]"

        python -m embedchain.bots.discord

        # or if you prefer to see the question and not only the answer, run it with
        python -m embedchain.bots.discord --include-question
        ```
    </Tab>
</Tabs>

### 🚀 Usage Instructions

- Go to the server where you have added your bot.
  ![Slash commands interaction with bot](https://github.com/embedchain/embedchain/assets/73601258/bf1414e3-d408-4863-b0d2-ef382a76467e)
- You can add data sources to the bot using the slash command:

/ec add <data_type> <url_or_text>

- You can ask your queries from the bot using the slash command:

/ec query <question>

- You can chat with the bot using the slash command:

/ec chat <question>

📝 Note: To use the bot privately, you can message the bot directly by right clicking the bot and selecting `Message`.

🎉 Happy Chatting! 🎉
"""
logger.info("### 🔑 Keys Setup")

logger.info("\n\n[DONE]", bright=True)