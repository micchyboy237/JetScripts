from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.document_loaders import UnstructuredURLLoader
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
# URL

This example covers how to load `HTML` documents from a list of `URLs` into the `Document` format that we can use downstream.

## Unstructured URL Loader

For the examples below, please install the `unstructured` library and see [this guide](/docs/integrations/providers/unstructured/) for more instructions on setting up Unstructured locally, including setting up required system dependencies:
"""
logger.info("# URL")

# %pip install --upgrade --quiet unstructured


urls = [
    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-8-2023",
    "https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-february-9-2023",
]

"""
Pass in ssl_verify=False with headers=headers to get past ssl_verification errors.
"""
logger.info("Pass in ssl_verify=False with headers=headers to get past ssl_verification errors.")

loader = UnstructuredURLLoader(urls=urls)

data = loader.load()

data[0]

"""
## Selenium URL Loader

This covers how to load HTML documents from a list of URLs using the `SeleniumURLLoader`.

Using `Selenium` allows us to load pages that require JavaScript to render.


To use the `SeleniumURLLoader`, you have to install `selenium` and `unstructured`.
"""
logger.info("## Selenium URL Loader")

# %pip install --upgrade --quiet selenium unstructured


urls = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://goo.gl/maps/NDSHwePEyaHMFGwh8",
]

loader = SeleniumURLLoader(urls=urls)

data = loader.load()

data[1]

"""
## Playwright URL Loader

>[Playwright](https://github.com/microsoft/playwright) is an open-source automation tool developed by `Microsoft` that allows you to programmatically control and automate web browsers. It is designed for end-to-end testing, scraping, and automating tasks across various web browsers such as `Chromium`, `Firefox`, and `WebKit`.

This covers how to load HTML documents from a list of URLs using the `PlaywrightURLLoader`.

[Playwright](https://playwright.dev/) enables reliable end-to-end testing for modern web apps.

As in the Selenium case, `Playwright` allows us to load and render the JavaScript pages.

To use the `PlaywrightURLLoader`, you have to install `playwright` and `unstructured`. Additionally, you have to install the `Playwright Chromium` browser:
"""
logger.info("## Playwright URL Loader")

# %pip install --upgrade --quiet playwright unstructured

# !playwright install

"""
Currently, nly the async method supported:
"""
logger.info("Currently, nly the async method supported:")


urls = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://goo.gl/maps/NDSHwePEyaHMFGwh8",
]

loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])

data = await loader.aload()
logger.success(format_json(data))

data[0]

logger.info("\n\n[DONE]", bright=True)