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
title: 🗑 delete
---

## Delete Document

`delete()` method allows you to delete a document previously added to the app.

### Usage
"""
logger.info("## Delete Document")


app = App()

forbes_doc_id = app.add("https://www.forbes.com/profile/elon-musk")
wiki_doc_id = app.add("https://en.wikipedia.org/wiki/Elon_Musk")

app.delete(forbes_doc_id)   # deletes the forbes document

"""
<Note>
    If you do not have the document id, you can use `app.db.get()` method to get the document and extract the `hash` key from `metadatas` dictionary object, which serves as the document id.
</Note>


## Delete Chat Session History

`delete_session_chat_history()` method allows you to delete all previous messages in a chat history.

### Usage
"""
logger.info("## Delete Chat Session History")


app = App()

app.add("https://www.forbes.com/profile/elon-musk")

app.chat("What is the net worth of Elon Musk?")

app.delete_session_chat_history()

"""
<Note>
    `delete_session_chat_history(session_id="session_1")` method also accepts `session_id` optional param for deleting chat history of a specific session.
    It assumes the default session if no `session_id` is provided.
</Note>
"""
logger.info("It assumes the default session if no `session_id` is provided.")

logger.info("\n\n[DONE]", bright=True)