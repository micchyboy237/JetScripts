from jet.logger import logger
from langchain_community.document_loaders import BiliBiliLoader
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
# BiliBili

>[Bilibili](https://www.bilibili.com/) is one of the most beloved long-form video sites in China.


This loader leverages the [bilibili-api](https://github.com/Nemo2011/bilibili-api) to retrieve text transcripts from `Bilibili` videos. To effectively use this loader, it's essential to have the `sessdata`, `bili_jct`, and `buvid3` cookie parameters. These can be obtained by logging into [Bilibili](https://www.bilibili.com/), then extracting the values of `sessdata`, `bili_jct`, and `buvid3` from the browser's developer tools.

If you choose to leave the cookie parameters blank, the Loader will still function, but it will only retrieve video information for the metadata and will not be able to fetch transcripts.

For detailed instructions on obtaining these credentials, refer to the guide [here](https://nemo2011.github.io/bilibili-api/#/get-credential).

The BiliBiliLoader provides a user-friendly interface for easily accessing transcripts of desired video content on Bilibili, making it an invaluable tool for those looking to analyze or utilize this media data.
"""
logger.info("# BiliBili")

# %pip install --upgrade --quiet  bilibili-api-python


SESSDATA = "<your sessdata>"
BUVID3 = "<your buvids>"
BILI_JCT = "<your bili_jct>"

loader = BiliBiliLoader(
    [
        "https://www.bilibili.com/video/BV1g84y1R7oE/",
    ],
    sessdata=SESSDATA,
    bili_jct=BILI_JCT,
    buvid3=BUVID3,
)

docs = loader.load()

docs

logger.info("\n\n[DONE]", bright=True)