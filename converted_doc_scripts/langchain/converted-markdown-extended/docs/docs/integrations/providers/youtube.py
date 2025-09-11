from jet.logger import logger
from langchain_community.document_loaders import GoogleApiYoutubeLoader
from langchain_community.document_loaders import YoutubeLoader
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

>[YouTube](https://www.youtube.com/) is an online video sharing and social media platform by Google.
> We download the `YouTube` transcripts and video information.

## Installation and Setup
"""
logger.info("# YouTube")

pip install youtube-transcript-api
pip install pytube

"""
See a [usage example](/docs/integrations/document_loaders/youtube_transcript).


## Document Loader

See a [usage example](/docs/integrations/document_loaders/youtube_transcript).
"""
logger.info("## Document Loader")


logger.info("\n\n[DONE]", bright=True)