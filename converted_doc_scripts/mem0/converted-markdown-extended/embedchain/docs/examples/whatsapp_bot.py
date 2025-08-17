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
title: '💬 WhatsApp Bot'
---

### 🚀 Getting started

1. Install embedchain python package:
"""
logger.info("### 🚀 Getting started")

pip install --upgrade embedchain

"""
2. Launch your WhatsApp bot:

<Tabs>
    <Tab title="docker">
        ```bash
#         docker run --name whatsapp-bot -e OPENAI_API_KEY=sk-xxx -p 8000:8000 embedchain/whatsapp-bot
        ```
    </Tab>
    <Tab title="python">
        ```bash
        python -m embedchain.bots.whatsapp --port 5000
        ```
    </Tab>
</Tabs>


If your bot needs to be accessible online, use your machine's public IP or DNS. Otherwise, employ a proxy server like [ngrok](https://ngrok.com/) to make your local bot accessible.

3. Create a free account on [Twilio](https://www.twilio.com/try-twilio)
    - Set up a WhatsApp Sandbox in your Twilio dashboard. Access it via the left sidebar: `Messaging > Try it out > Send a WhatsApp Message`.
    - Follow on-screen instructions to link a phone number for chatting with your bot
    - Copy your bot's public URL, add /chat at the end, and paste it in Twilio's WhatsApp Sandbox settings under "When a message comes in". Save the settings.

- Copy your bot's public url, append `/chat` at the end and paste it under `When a message comes in` under the `Sandbox settings` for Whatsapp in Twilio. Save your settings.

### 💬 How to use

- To connect a new number or reconnect an old one in the Sandbox, follow Twilio's instructions.
- To include data sources, use this command:

add <url_or_text>

- To ask the bot questions, just type your query:

<your-question-here>

### Example

Here is an example of Elon Musk WhatsApp Bot that we created:

<img src="/images/whatsapp.jpg"/>
"""
logger.info("### 💬 How to use")

logger.info("\n\n[DONE]", bright=True)