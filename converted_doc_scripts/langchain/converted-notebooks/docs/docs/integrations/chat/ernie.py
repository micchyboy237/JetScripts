from jet.logger import logger
from langchain_community.chat_models import ErnieBotChat
from langchain_community.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint
from langchain_core.messages import HumanMessage
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
sidebar_label: Ernie Bot Chat
---

# ErnieBotChat

[ERNIE-Bot](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/jlil56u11) is a large language model developed by Baidu, covering a huge amount of Chinese data.
This notebook covers how to get started with ErnieBot chat models.

**Deprecated Warning**

We recommend users switch from `langchain_community.chat_models.ErnieBotChat` to `langchain_community.chat_models.QianfanChatEndpoint`.

documentation for `QianfanChatEndpoint` is [here](/docs/integrations/chat/baidu_qianfan_endpoint/).

There are 4 reasons why we recommend users to use `QianfanChatEndpoint`:

1. `QianfanChatEndpoint` supports more LLMs in the Qianfan platform.
2. `QianfanChatEndpoint` supports streaming mode.
3. `QianfanChatEndpoint` support function calling usage.
4. `ErnieBotChat` is no longer maintained and has been deprecated.

Some tips for migration:

- change `ernie_client_id` to `qianfan_ak`, also change `ernie_client_secret` to `qianfan_sk`.
- install `qianfan` package. like `pip install qianfan`
- change `ErnieBotChat` to `QianfanChatEndpoint`.
"""
logger.info("# ErnieBotChat")


chat = QianfanChatEndpoint(
    qianfan_ak="your qianfan ak",
    qianfan_sk="your qianfan sk",
)

"""
## Usage
"""
logger.info("## Usage")


chat = ErnieBotChat(
    ernie_client_id="YOUR_CLIENT_ID", ernie_client_secret="YOUR_CLIENT_SECRET"
)

"""
or you can set `client_id` and `client_secret` in your environment variables
```bash
export ERNIE_CLIENT_ID=YOUR_CLIENT_ID
export ERNIE_CLIENT_SECRET=YOUR_CLIENT_SECRET
```
"""
logger.info("or you can set `client_id` and `client_secret` in your environment variables")

chat([HumanMessage(content="hello there, who are you?")])

logger.info("\n\n[DONE]", bright=True)