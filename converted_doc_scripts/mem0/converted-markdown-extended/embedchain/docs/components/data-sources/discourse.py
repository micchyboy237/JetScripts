from embedchain.chunkers.discourse import DiscourseChunker
from embedchain.config.add_config import ChunkerConfig
from embedchain.loaders.discourse import DiscourseLoader
from embedchain.pipeline import Pipeline as App
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
title: '🗨️ Discourse'
---

You can now easily load data from your community built with [Discourse](https://discourse.org/).

## Example

1. Setup the Discourse Loader with your community url.
"""
logger.info("## Example")


dicourse_loader = DiscourseLoader(config={"domain": "https://community.openai.com"})

"""
2. Once you setup the loader, you can create an app and load data using the above discourse loader
"""
logger.info("2. Once you setup the loader, you can create an app and load data using the above discourse loader")


# os.environ["OPENAI_API_KEY"] = "sk-xxx"

app = App()

app.add("openai after:2023-10-1", data_type="discourse", loader=dicourse_loader)

question = "Where can I find the MLX API status page?"
app.query(question)

"""
NOTE: The `add` function of the app will accept any executable search query to load data. Refer [Discourse API Docs](https://docs.discourse.org/#tag/Search) to learn more about search queries.

3. We automatically create a chunker to chunk your discourse data, however if you wish to provide your own chunker class. Here is how you can do that:
"""
logger.info("NOTE: The `add` function of the app will accept any executable search query to load data. Refer [Discourse API Docs](https://docs.discourse.org/#tag/Search) to learn more about search queries.")


discourse_chunker_config = ChunkerConfig(chunk_size=1000, chunk_overlap=0, length_function=len)
discourse_chunker = DiscourseChunker(config=discourse_chunker_config)

app.add("openai", data_type='discourse', loader=dicourse_loader, chunker=discourse_chunker)

logger.info("\n\n[DONE]", bright=True)