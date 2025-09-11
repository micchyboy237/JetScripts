from jet.logger import logger
from langchain_community.chat_models import ChatBaichuan
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
sidebar_label: Baichuan Chat
---

# Chat with Baichuan-192K

Baichuan chat models API by Baichuan Intelligent Technology. For more information, see [https://platform.baichuan-ai.com/docs/api](https://platform.baichuan-ai.com/docs/api)
"""
logger.info("# Chat with Baichuan-192K")


chat = ChatBaichuan(baichuan_)

"""
Alternatively, you can set your API key with:
"""
logger.info("Alternatively, you can set your API key with:")


os.environ["BAICHUAN_API_KEY"] = "YOUR_API_KEY"

chat([HumanMessage(content="我日薪8块钱，请问在闰年的二月，我月薪多少")])

"""
## Chat with Baichuan-192K with Streaming
"""
logger.info("## Chat with Baichuan-192K with Streaming")

chat = ChatBaichuan(
    baichuan_streaming=True,
)

chat([HumanMessage(content="我日薪8块钱，请问在闰年的二月，我月薪多少")])

logger.info("\n\n[DONE]", bright=True)