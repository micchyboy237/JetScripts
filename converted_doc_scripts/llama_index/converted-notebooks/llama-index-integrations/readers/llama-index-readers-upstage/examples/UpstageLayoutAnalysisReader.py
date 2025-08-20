from jet.logger import CustomLogger
from llama_index.readers.upstage import UpstageLayoutAnalysisReader
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# UpstageLayoutAnalysisReader

This notebook covers how to get started with `UpstageLayoutAnalysisReader`.

## Installation
Please install `llama_index` package.
```bash
pip install llama-index-readers-upstage
```

## Environment Setup
Make sure to set the following environment variables:
- `UPSTAGE_API_KEY`: Your Upstage API key. Read [Upstage console](https://console.upstage.ai/) to get your API key.

### Usage
"""
logger.info("# UpstageLayoutAnalysisReader")


os.environ["UPSTAGE_API_KEY"] = "YOUR_API_KEY"


file_path = "/PATH/TO/YOUR/FILE.pdf"

reader = UpstageLayoutAnalysisReader()

docs = reader.load_data(file_path=file_path, split="element")

for doc in docs[:3]:
    logger.debug(doc)

logger.info("\n\n[DONE]", bright=True)