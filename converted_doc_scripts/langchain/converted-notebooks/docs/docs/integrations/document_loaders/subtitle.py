from jet.logger import logger
from langchain_community.document_loaders import SRTLoader
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
# Subtitle

>[The SubRip file format](https://en.wikipedia.org/wiki/SubRip#SubRip_file_format) is described on the `Matroska` multimedia container format website as "perhaps the most basic of all subtitle formats." `SubRip (SubRip Text)` files are named with the extension `.srt`, and contain formatted lines of plain text in groups separated by a blank line. Subtitles are numbered sequentially, starting at 1. The timecode format used is hours:minutes:seconds,milliseconds with time units fixed to two zero-padded digits and fractions fixed to three zero-padded digits (00:00:00,000). The fractional separator used is the comma, since the program was written in France.

How to load data from subtitle (`.srt`) files

Please, download the [example .srt file from here](https://www.opensubtitles.org/en/subtitles/5575150/star-wars-the-clone-wars-crisis-at-the-heart-en).
"""
logger.info("# Subtitle")

# %pip install --upgrade --quiet  pysrt


loader = SRTLoader(
    "example_data/Star_Wars_The_Clone_Wars_S06E07_Crisis_at_the_Heart.srt"
)

docs = loader.load()

docs[0].page_content[:100]

logger.info("\n\n[DONE]", bright=True)