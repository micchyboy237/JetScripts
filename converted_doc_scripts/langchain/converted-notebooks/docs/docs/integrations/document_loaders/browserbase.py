from jet.logger import logger
from langchain_community.document_loaders import BrowserbaseLoader
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
# Browserbase

[Browserbase](https://browserbase.com) is a developer platform to reliably run, manage, and monitor headless browsers.

Power your AI data retrievals with:
- [Serverless Infrastructure](https://docs.browserbase.com/under-the-hood) providing reliable browsers to extract data from complex UIs
- [Stealth Mode](https://docs.browserbase.com/features/stealth-mode) with included fingerprinting tactics and automatic captcha solving
- [Session Debugger](https://docs.browserbase.com/features/sessions) to inspect your Browser Session with networks timeline and logs
- [Live Debug](https://docs.browserbase.com/guides/session-debug-connection/browser-remote-control) to quickly debug your automation

## Installation and Setup

- Get an API key and Project ID from [browserbase.com](https://browserbase.com) and set it in environment variables (`BROWSERBASE_API_KEY`, `BROWSERBASE_PROJECT_ID`).
- Install the [Browserbase SDK](http://github.com/browserbase/python-sdk):
"""
logger.info("# Browserbase")

# %pip install browserbase

"""
## Loading documents

You can load webpages into LangChain using `BrowserbaseLoader`. Optionally, you can set `text_content` parameter to convert the pages to text-only representation.
"""
logger.info("## Loading documents")



load_dotenv()

BROWSERBASE_API_KEY = os.getenv("BROWSERBASE_API_KEY")
BROWSERBASE_PROJECT_ID = os.getenv("BROWSERBASE_PROJECT_ID")

loader = BrowserbaseLoader(
    api_key=BROWSERBASE_API_KEY,
    project_id=BROWSERBASE_PROJECT_ID,
    urls=[
        "https://example.com",
    ],
    text_content=False,
)

docs = loader.load()
logger.debug(docs[0].page_content[:61])

"""
### Loader Options

- `urls` Required. A list of URLs to fetch.
- `text_content` Retrieve only text content. Default is `False`.
- `api_key` Browserbase API key. Default is `BROWSERBASE_API_KEY` env variable.
- `project_id` Browserbase Project ID. Default is `BROWSERBASE_PROJECT_ID` env variable.
- `session_id` Optional. Provide an existing Session ID.
- `proxy` Optional. Enable/Disable Proxies.
"""
logger.info("### Loader Options")

logger.info("\n\n[DONE]", bright=True)