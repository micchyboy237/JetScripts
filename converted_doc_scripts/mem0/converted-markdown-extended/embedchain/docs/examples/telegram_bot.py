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
title: "📱 Telegram Bot"
---

### 🖼️ Template Setup

- Open the Telegram app and search for the `BotFather` user.
- Start a chat with BotFather and use the `/newbot` command to create a new bot.
- Follow the instructions to choose a name and username for your bot.
- Once the bot is created, BotFather will provide you with a unique token for your bot.

<Tabs>
    <Tab title="docker">
        ```bash
#         docker run --name telegram-bot -e OPENAI_API_KEY=sk-xxx -e TELEGRAM_BOT_TOKEN=xxx -p 8000:8000 embedchain/telegram-bot
        ```

    <Note>
    If you wish to use **Docker**, you would need to host your bot on a server.
    You can use [ngrok](https://ngrok.com/) to expose your localhost to the
    internet and then set the webhook using the ngrok URL.
    </Note>

    </Tab>
    <Tab title="replit">
    <Card>
        Fork <ins>**[this](https://replit.com/@taranjeetio/EC-Telegram-Bot-Template?v=1#README.md)**</ins> replit template.
    </Card>

#     - Set your `OPENAI_API_KEY` in Secrets.
    - Set the unique token as `TELEGRAM_BOT_TOKEN` in Secrets.

    </Tab>

</Tabs>

- Click on `Run` in the replit container and a URL will get generated for your bot.
- Now set your webhook by running the following link in your browser:
"""
logger.info("### 🖼️ Template Setup")

https://api.telegram.org/bot<Your_Telegram_Bot_Token>/setWebhook?url=<Replit_Generated_URL>

"""
- When you get a successful response in your browser, your bot is ready to be used.

### 🚀 Usage Instructions

- Open your bot by searching for it using the bot name or bot username.
- Click on `Start` or type `/start` and follow the on screen instructions.

🎉 Happy Chatting! 🎉
"""
logger.info("### 🚀 Usage Instructions")

logger.info("\n\n[DONE]", bright=True)