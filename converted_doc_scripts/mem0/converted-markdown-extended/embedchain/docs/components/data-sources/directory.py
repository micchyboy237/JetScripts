from embedchain import App
from embedchain.loaders.directory_loader import DirectoryLoader
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
title: '📁 Directory/Folder'
---

To use an entire directory as data source, just add `data_type` as `directory` and pass in the path of the local directory.

### Without customization
"""
logger.info("### Without customization")


# os.environ["OPENAI_API_KEY"] = "sk-xxx"

app = App()
app.add("./elon-musk", data_type="directory")
response = app.query("list all files")
logger.debug(response)

"""
### Customization
"""
logger.info("### Customization")


# os.environ["OPENAI_API_KEY"] = "sk-xxx"
lconfig = {
    "recursive": True,
    "extensions": [".txt"]
}
loader = DirectoryLoader(config=lconfig)
app = App()
app.add("./elon-musk", loader=loader)
response = app.query("what are all the files related to?")
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)