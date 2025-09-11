from jet.logger import logger
from langchain_community.tools import YouTubeSearchTool
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
# YouTube

>[YouTube Search](https://github.com/joetats/youtube_search) package searches `YouTube` videos avoiding using their heavily rate-limited API.
>
>It uses the form on the `YouTube` homepage and scrapes the resulting page.

This notebook shows how to use a tool to search YouTube.

Adapted from [https://github.com/venuv/langchain_yt_tools](https://github.com/venuv/langchain_yt_tools)
"""
logger.info("# YouTube")

# %pip install --upgrade --quiet  youtube_search


tool = YouTubeSearchTool()

tool.run("lex fridman")

"""
You can also specify the number of results that are returned
"""
logger.info("You can also specify the number of results that are returned")

tool.run("lex friedman,5")

logger.info("\n\n[DONE]", bright=True)