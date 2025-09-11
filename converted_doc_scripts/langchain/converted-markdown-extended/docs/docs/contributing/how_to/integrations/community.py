from jet.logger import logger
from langchain_community.chat_models import ChatParrotLink
from langchain_community.llms import ParrotLinkLLM
from langchain_community.vectorstores import ParrotLinkVectorStore
from langchain_core.language_models.chat_models import BaseChatModel
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
pagination_next: null
pagination_prev: null
---
## How to add a community integration (not recommended)

:::danger

We recommend following the [main integration guide](./index.mdx) to add new integrations instead.

If you follow this guide, there is a high likelihood we will close your PR with the above
guide linked without much discussion.

:::

The `langchain-community` package is in `libs/community`.

It can be installed with `pip install langchain-community`, and exported members can be imported with code like
"""
logger.info("## How to add a community integration (not recommended)")


"""
The `community` package relies on manually-installed dependent packages, so you will see errors
if you try to import a package that is not installed. In our fake example, if you tried to import `ParrotLinkLLM` without installing `parrot-link-sdk`, you will see an `ImportError` telling you to install it when trying to use it.

Let's say we wanted to implement a chat model for Parrot Link AI. We would create a new file in `libs/community/langchain_community/chat_models/parrot_link.py` with the following code:
"""
logger.info("The `community` package relies on manually-installed dependent packages, so you will see errors")


class ChatParrotLink(BaseChatModel):
    """ChatParrotLink chat model.

    Example:
        .. code-block:: python


            model = ChatParrotLink()
    """

    ...

"""
And we would write tests in:

- Unit tests: `libs/community/tests/unit_tests/chat_models/test_parrot_link.py`
- Integration tests: `libs/community/tests/integration_tests/chat_models/test_parrot_link.py`

And add documentation to:

- `docs/docs/integrations/chat/parrot_link.ipynb`
"""
logger.info("And we would write tests in:")

logger.info("\n\n[DONE]", bright=True)