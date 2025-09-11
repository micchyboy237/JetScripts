from jet.logger import logger
from langchain_taiga.toolkits import TaigaToolkit
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
# Taiga

> [Taiga](https://docs.taiga.io/) is an open-source project management platform designed for agile teams, offering features like Kanban, Scrum, and issue tracking.

## Installation and Setup

Install the `langchain-taiga` package:
"""
logger.info("# Taiga")

pip install langchain-taiga

"""
You must provide a logins via environment variable so the tools can authenticate.
"""
logger.info("You must provide a logins via environment variable so the tools can authenticate.")

export TAIGA_URL="https://taiga.xyz.org/"
export TAIGA_API_URL="https://taiga.xyz.org/"
export TAIGA_USERNAME="username"
export TAIGA_PASSWORD="pw"
# export OPENAI_API_KEY="OPENAI_API_KEY"

"""
---

## Tools

See a [usage example](/docs/integrations/tools/taiga)

---

## Toolkit

`TaigaToolkit` groups multiple Taiga-related tools into a single interface.
"""
logger.info("## Tools")


toolkit = TaigaToolkit()
tools = toolkit.get_tools()

"""
---

## Future Integrations


Check the [Taiga Developer Docs](https://docs.taiga.io/) for more information, and watch for updates or advanced usage examples in the [langchain_taiga GitHub repo](https://github.com/Shikenso-Analytics/langchain-taiga).
"""
logger.info("## Future Integrations")

logger.info("\n\n[DONE]", bright=True)