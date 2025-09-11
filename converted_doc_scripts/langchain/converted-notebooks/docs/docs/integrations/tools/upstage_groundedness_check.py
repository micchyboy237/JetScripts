from jet.logger import logger
from langchain_upstage import UpstageGroundednessCheck
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
---
sidebar_label: Upstage
---

# Upstage Groundedness Check

This notebook covers how to get started with Upstage groundedness check models.

## Installation  

Install `langchain-upstage` package.

```bash
pip install -U langchain-upstage
```

## Environment Setup

Make sure to set the following environment variables:

- `UPSTAGE_API_KEY`: Your Upstage API key from [Upstage developers document](https://developers.upstage.ai/docs/getting-started/quick-start).
"""
logger.info("# Upstage Groundedness Check")


os.environ["UPSTAGE_API_KEY"] = "YOUR_API_KEY"

"""
## Usage

Initialize `UpstageGroundednessCheck` class.
"""
logger.info("## Usage")


groundedness_check = UpstageGroundednessCheck()

"""
Use the `run` method to check the groundedness of the input text.
"""
logger.info("Use the `run` method to check the groundedness of the input text.")

request_input = {
    "context": "Mauna Kea is an inactive volcano on the island of Hawai'i. Its peak is 4,207.3 m above sea level, making it the highest point in Hawaii and second-highest peak of an island on Earth.",
    "answer": "Mauna Kea is 5,207.3 meters tall.",
}

response = groundedness_check.invoke(request_input)
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)