from embedchain import App
from embedchain.config import AppConfig
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
## Embedchain chromadb server example

This notebook shows an example of how you can use embedchain with chromdb (server). 


First, run chroma inside docker using the following command:


```bash
git clone https://github.com/chroma-core/chroma
cd chroma && docker-compose up -d --build
```
"""
logger.info("## Embedchain chromadb server example")



chromadb_host = "localhost"
chromadb_port = 8000

config = AppConfig(host=chromadb_host, port=chromadb_port)
elon_bot = App(config)

elon_bot.add("web_page", "https://en.wikipedia.org/wiki/Elon_Musk")
elon_bot.add("web_page", "https://www.tesla.com/elon-musk")

elon_bot.query("How many companies does Elon Musk run?")

logger.info("\n\n[DONE]", bright=True)