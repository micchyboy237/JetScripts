from jet.logger import logger
from langchain_community.document_loaders import MastodonTootsLoader
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
# Mastodon

>[Mastodon](https://joinmastodon.org/) is a federated social media and social networking service.

This loader fetches the text from the "toots" of a list of `Mastodon` accounts, using the `Mastodon.py` Python package.

Public accounts can the queried by default without any authentication. If non-public accounts or instances are queried, you have to register an application for your account which gets you an access token, and set that token and your account's API base URL.

Then you need to pass in the Mastodon account names you want to extract, in the `@account@instance` format.
"""
logger.info("# Mastodon")


# %pip install --upgrade --quiet  Mastodon.py

loader = MastodonTootsLoader(
    mastodon_accounts=["@Gargron@mastodon.social"],
    number_toots=50,  # Default value is 100
)

documents = loader.load()
for doc in documents[:3]:
    logger.debug(doc.page_content)
    logger.debug("=" * 80)

"""
The toot texts (the documents' `page_content`) is by default HTML as returned by the Mastodon API.
"""
logger.info("The toot texts (the documents' `page_content`) is by default HTML as returned by the Mastodon API.")

logger.info("\n\n[DONE]", bright=True)