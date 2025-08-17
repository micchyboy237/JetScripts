from embedchain.bots import PoeBot
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
title: '🔮 Poe Bot'
---

### 🚀 Getting started

1. Install embedchain python package:
"""
logger.info("### 🚀 Getting started")

pip install fastapi-poe==0.0.16

"""
2. Create a free account on [Poe](https://www.poe.com?utm_source=embedchain).
3. Click "Create Bot" button on top left.
4. Give it a handle and an optional description.
5. Select `Use API`.
6. Under `API URL` enter your server or ngrok address. You can use your machine's public IP or DNS. Otherwise, employ a proxy server like [ngrok](https://ngrok.com/) to make your local bot accessible.
7. Copy your api key and paste it in `.env` as `POE_API_KEY`.
# 8. You will need to set `OPENAI_API_KEY` for generating embeddings and using LLM. Copy your MLX API key from [here](https://platform.openai.com/account/api-keys) and paste it in `.env` as `OPENAI_API_KEY`.
9. Now create your bot using the following code snippet.
"""
logger.info("2. Create a free account on [Poe](https://www.poe.com?utm_source=embedchain).")


poe_bot = PoeBot()

poe_bot.add("https://en.wikipedia.org/wiki/Adam_D%27Angelo")
poe_bot.add("https://www.youtube.com/watch?v=pJQVAqmKua8")

poe_bot.start()

"""
10. You can paste the above in a file called `your_script.py` and then simply do
"""
logger.info("10. You can paste the above in a file called `your_script.py` and then simply do")

python your_script.py

"""
Now your bot will start running at port `8080` by default.

11. You can refer the [Supported Data formats](https://docs.embedchain.ai/advanced/data_types) section to refer the supported data types in embedchain.

12. Click `Run check` to make sure your machine can be reached.
13. Make sure your bot is private if that's what you want.
14. Click `Create bot` at the bottom to finally create the bot
15. Now your bot is created.

### 💬 How to use

- To ask the bot questions, just type your query in the Poe interface:

<your-question-here>

- If you wish to add more data source to the bot, simply update your script and add as many `.add` as you like. You need to restart the server.
"""
logger.info("### 💬 How to use")

logger.info("\n\n[DONE]", bright=True)