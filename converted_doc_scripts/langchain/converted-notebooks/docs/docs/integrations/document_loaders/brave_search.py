from jet.logger import logger
from langchain_community.document_loaders import BraveSearchLoader
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
# Brave Search


>[Brave Search](https://en.wikipedia.org/wiki/Brave_Search) is a search engine developed by Brave Software.
> - `Brave Search` uses its own web index. As of May 2022, it covered over 10 billion pages and was used to serve 92% 
> of search results without relying on any third-parties, with the remainder being retrieved 
> server-side from the Bing API or (on an opt-in basis) client-side from Google. According 
> to Brave, the index was kept "intentionally smaller than that of Google or Bing" in order to 
> help avoid spam and other low-quality content, with the disadvantage that "Brave Search is 
> not yet as good as Google in recovering long-tail queries."
>- `Brave Search Premium`: As of April 2023 Brave Search is an ad-free website, but it will 
> eventually switch to a new model that will include ads and premium users will get an ad-free experience.
> User data including IP addresses won't be collected from its users by default. A premium account 
> will be required for opt-in data-collection.

## Installation and Setup

To get access to the Brave Search API, you need to [create an account and get an API key](https://api.search.brave.com/app/dashboard).
"""
logger.info("# Brave Search")




"""
## Example
"""
logger.info("## Example")

loader = BraveSearchLoader(
    query="obama middle name", api_key=api_key, search_kwargs={"count": 3}
)
docs = loader.load()
len(docs)

[doc.metadata for doc in docs]

[doc.page_content for doc in docs]

logger.info("\n\n[DONE]", bright=True)