from jet.logger import logger
from langchain_community.chat_loaders.facebook_messenger import (
FolderFacebookMessengerChatLoader,
SingleFileFacebookMessengerChatLoader,
)
from langchain_community.chat_loaders.whatsapp import WhatsAppChatLoader
from langchain_community.document_loaders import FacebookChatLoader
from langchain_community.embeddings.laser import LaserEmbeddings
from langchain_community.vectorstores import FAISS
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
# Facebook - Meta

>[Meta Platforms, Inc.](https://www.facebook.com/), doing business as `Meta`, formerly
> named `Facebook, Inc.`, and `TheFacebook, Inc.`, is an American multinational technology
> conglomerate. The company owns and operates `Facebook`, `Instagram`, `Threads`,
> and `WhatsApp`, among other products and services.

## Embedding models

### LASER

>[LASER](https://github.com/facebookresearch/LASER) is a Python library developed by
> the `Meta AI Research` team and used for
> creating multilingual sentence embeddings for
> [over 147 languages as of 2/25/2024](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200)
"""
logger.info("# Facebook - Meta")

pip install laser_encoders

"""
See a [usage example](/docs/integrations/text_embedding/laser).
"""
logger.info("See a [usage example](/docs/integrations/text_embedding/laser).")


"""
## Document loaders

### Facebook Messenger

>[Messenger](https://en.wikipedia.org/wiki/Messenger_(software)) is an instant messaging app and
> platform developed by `Meta Platforms`. Originally developed as `Facebook Chat` in 2008, the company revamped its
> messaging service in 2010.

See a [usage example](/docs/integrations/document_loaders/facebook_chat).
"""
logger.info("## Document loaders")


"""
## Vector stores

### Facebook Faiss

>[Facebook AI Similarity Search (Faiss)](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
> is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that
> search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting
> code for evaluation and parameter tuning.

[Faiss documentation](https://faiss.ai/).

We need to install `faiss` python package.
"""
logger.info("## Vector stores")

pip install faiss-gpu # For CUDA 7.5+ supported GPU's.

"""
OR
"""
logger.info("OR")

pip install faiss-cpu # For CPU Installation

"""
See a [usage example](/docs/integrations/vectorstores/faiss).
"""
logger.info("See a [usage example](/docs/integrations/vectorstores/faiss).")


"""
## Chat loaders

### Facebook Messenger

>[Messenger](https://en.wikipedia.org/wiki/Messenger_(software)) is an instant messaging app and
> platform developed by `Meta Platforms`. Originally developed as `Facebook Chat` in 2008, the company revamped its
> messaging service in 2010.

See a [usage example](/docs/integrations/chat_loaders/facebook).
"""
logger.info("## Chat loaders")


"""
### Facebook WhatsApp

See a [usage example](/docs/integrations/chat_loaders/whatsapp).
"""
logger.info("### Facebook WhatsApp")


logger.info("\n\n[DONE]", bright=True)