from jet.logger import logger
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
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
# Streamlit

>[Streamlit](https://streamlit.io/) is a faster way to build and share data apps.
>`Streamlit` turns data scripts into shareable web apps in minutes. All in pure Python. No front-end experience required.
>See more examples at [streamlit.io/generative-ai](https://streamlit.io/generative-ai).

## Installation and Setup

We need to install the  `streamlit` Python package:
"""
logger.info("# Streamlit")

pip install streamlit

"""
## Memory

See a [usage example](/docs/integrations/memory/streamlit_chat_message_history).
"""
logger.info("## Memory")


"""
## Callbacks

See a [usage example](/docs/integrations/callbacks/streamlit).
"""
logger.info("## Callbacks")


logger.info("\n\n[DONE]", bright=True)