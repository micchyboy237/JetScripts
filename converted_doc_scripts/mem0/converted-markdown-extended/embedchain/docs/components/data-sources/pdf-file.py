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
title: '📰 PDF'
---

You can load any pdf file from your local file system or through a URL.

## Usage

### Load from a local file
"""
logger.info("## Usage")

app = App()
app.add('/path/to/file.pdf', data_type='pdf_file')

"""
### Load from URL
"""
logger.info("### Load from URL")

app = App()
app.add('https://arxiv.org/pdf/1706.03762.pdf', data_type='pdf_file')
app.query("What is the paper 'attention is all you need' about?", citations=True)

"""
We also store the page number under the key `page` with each chunk that helps understand where the answer is coming from. You can fetch the `page` key while during retrieval (refer to the example given above).

<Note>
Note that we do not support password protected pdf files.
</Note>
"""
logger.info("We also store the page number under the key `page` with each chunk that helps understand where the answer is coming from. You can fetch the `page` key while during retrieval (refer to the example given above).")

logger.info("\n\n[DONE]", bright=True)