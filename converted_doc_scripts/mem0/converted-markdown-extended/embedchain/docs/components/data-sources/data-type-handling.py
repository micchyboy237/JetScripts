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
title: 'Data type handling'
---

## Automatic data type detection

The add method automatically tries to detect the data_type, based on your input for the source argument. So `app.add('https://www.youtube.com/watch?v=dQw4w9WgXcQ')` is enough to embed a YouTube video.

This detection is implemented for all formats. It is based on factors such as whether it's a URL, a local file, the source data type, etc.

### Debugging automatic detection

Set `log_level: DEBUG` in the config yaml to debug if the data type detection is done right or not. Otherwise, you will not know when, for instance, an invalid filepath is interpreted as raw text instead.

### Forcing a data type

To omit any issues with the data type detection, you can **force** a data_type by adding it as a `add` method argument.
The examples below show you the keyword to force the respective `data_type`.

Forcing can also be used for edge cases, such as interpreting a sitemap as a web_page, for reading its raw text instead of following links.

## Remote data types

<Tip>
**Use local files in remote data types**

Some data_types are meant for remote content and only work with URLs.
You can pass local files by formatting the path using the `file:` [URI scheme](https://en.wikipedia.org/wiki/File_URI_scheme), e.g. `file:///info.pdf`.
</Tip>

## Reusing a vector database

Default behavior is to create a persistent vector db in the directory **./db**. You can split your application into two Python scripts: one to create a local vector db and the other to reuse this local persistent vector db. This is useful when you want to index hundreds of documents and separately implement a chat interface.

Create a local index:
"""
logger.info("## Automatic data type detection")


config = {
    "app": {
        "config": {
            "id": "app-1"
        }
    }
}
naval_chat_bot = App.from_config(config=config)
naval_chat_bot.add("https://www.youtube.com/watch?v=3qHkcs3kG44")
naval_chat_bot.add("https://navalmanack.s3.amazonaws.com/Eric-Jorgenson_The-Almanack-of-Naval-Ravikant_Final.pdf")

"""
You can reuse the local index with the same code, but without adding new documents:
"""
logger.info("You can reuse the local index with the same code, but without adding new documents:")


config = {
    "app": {
        "config": {
            "id": "app-1"
        }
    }
}
naval_chat_bot = App.from_config(config=config)
logger.debug(naval_chat_bot.query("What unique capacity does Naval argue humans possess when it comes to understanding explanations or concepts?"))

"""
## Resetting an app and vector database

You can reset the app by simply calling the `reset` method. This will delete the vector database and all other app related files.
"""
logger.info("## Resetting an app and vector database")


app = App()config = {
    "app": {
        "config": {
            "id": "app-1"
        }
    }
}
naval_chat_bot = App.from_config(config=config)
app.add("https://www.youtube.com/watch?v=3qHkcs3kG44")
app.reset()

logger.info("\n\n[DONE]", bright=True)