from jet.logger import logger
from langchain_community.chat_models.solar import SolarChat
from langchain_core.messages import HumanMessage, SystemMessage
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
Deprecated since version 0.0.34: Use langchain_upstage.ChatUpstage instead.
"""
logger.info("Deprecated since version 0.0.34: Use langchain_upstage.ChatUpstage instead.")


os.environ["SOLAR_API_KEY"] = "SOLAR_API_KEY"


chat = SolarChat(max_tokens=1024)

messages = [
    SystemMessage(
        content="You are a helpful assistant who translates English to Korean."
    ),
    HumanMessage(
        content="Translate this sentence from English to Korean. I want to build a project of large language model."
    ),
]

chat.invoke(messages)

logger.info("\n\n[DONE]", bright=True)