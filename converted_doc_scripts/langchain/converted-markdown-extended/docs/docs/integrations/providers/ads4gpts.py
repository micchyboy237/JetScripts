from ads4gpts_langchain import Ads4gptsInlineBannerTool
from ads4gpts_langchain import Ads4gptsInlineConversationalTool
from ads4gpts_langchain import Ads4gptsInlineSponsoredResponseTool
from ads4gpts_langchain import Ads4gptsSuggestedBannerTool
from ads4gpts_langchain import Ads4gptsSuggestedPromptTool
from ads4gpts_langchain import Ads4gptsToolkit
from jet.logger import logger
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
# ADS4GPTs

> [ADS4GPTs](https://www.ads4gpts.com/) is building the open monetization backbone of the AI-Native internet. It helps AI applications monetize through advertising with a UX and Privacy first approach.

## Installation and Setup

### Using pip
You can install the package directly from PyPI:
"""
logger.info("# ADS4GPTs")

pip install ads4gpts-langchain

"""
### From Source
Alternatively, install from source:
"""
logger.info("### From Source")

git clone https://github.com/ADS4GPTs/ads4gpts.git
cd ads4gpts/libs/python-sdk/ads4gpts-langchain
pip install .

"""
## Prerequisites

- Python 3.11+
- ADS4GPTs API Key ([Obtain API Key](https://www.ads4gpts.com))

## Environment Variables
Set the following environment variables for API authentication:
"""
logger.info("## Prerequisites")

export ADS4GPTS_API_KEY='your-ads4gpts-api-key'

"""
Alternatively, API keys can be passed directly when initializing classes or stored in a `.env` file.

## Tools

ADS4GPTs provides two main tools for monetization:

### Ads4gptsInlineSponsoredResponseTool
This tool fetches native, sponsored responses that can be seamlessly integrated within your AI application's outputs.
"""
logger.info("## Tools")


"""
### Ads4gptsSuggestedPromptTool
Generates sponsored prompt suggestions to enhance user engagement and provide monetization opportunities.
"""
logger.info("### Ads4gptsSuggestedPromptTool")


"""
### Ads4gptsInlineConversationalTool
Delivers conversational sponsored content that naturally fits within chat interfaces and dialogs.
"""
logger.info("### Ads4gptsInlineConversationalTool")


"""
### Ads4gptsInlineBannerTool
Provides inline banner advertisements that can be displayed within your AI application's response.
"""
logger.info("### Ads4gptsInlineBannerTool")


"""
### Ads4gptsSuggestedBannerTool
Generates banner advertisement suggestions that can be presented to users as recommended content.
"""
logger.info("### Ads4gptsSuggestedBannerTool")


"""
## Toolkit

The `Ads4gptsToolkit` combines these tools for convenient access in LangChain applications.
"""
logger.info("## Toolkit")


logger.info("\n\n[DONE]", bright=True)