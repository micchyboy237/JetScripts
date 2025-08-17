from embedchain import App
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: '📊 CSV'
---

You can load any csv file from your local file system or through a URL. Headers are included for each line, so if you have an `age` column, `18` will be added as `age: 18`.

## Usage

### Load from a local file
"""
logger.info("## Usage")

app = App()
app.add('/path/to/file.csv', data_type='csv')

"""
### Load from URL
"""
logger.info("### Load from URL")

app = App()
app.add('https://people.sc.fsu.edu/~jburkardt/data/csv/airtravel.csv', data_type="csv")

"""
<Note>
There is a size limit allowed for csv file beyond which it can throw error. This limit is set by the LLMs. Please consider chunking large csv files into smaller csv files.
</Note>
"""
logger.info("There is a size limit allowed for csv file beyond which it can throw error. This limit is set by the LLMs. Please consider chunking large csv files into smaller csv files.")

logger.info("\n\n[DONE]", bright=True)