from jet.logger import logger
from langchain_community.chat_models import ChatTongyi
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
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
sidebar_label: Tongyi Qwen
---

# ChatTongyi
Tongyi Qwen is a large language model developed by Alibaba's Damo Academy. It is capable of understanding user intent through natural language understanding and semantic analysis, based on user input in natural language. It provides services and assistance to users in different domains and tasks. By providing clear and detailed instructions, you can obtain results that better align with your expectations.
In this notebook, we will introduce how to use langchain with [Tongyi](https://www.aliyun.com/product/dashscope) mainly in `Chat` corresponding
 to the package `langchain/chat_models` in langchain
"""
logger.info("# ChatTongyi")

# %pip install --upgrade --quiet  dashscope

# from getpass import getpass

# DASHSCOPE_API_KEY = getpass()


os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY


chatLLM = ChatTongyi(
    streaming=True,
)
res = chatLLM.stream([HumanMessage(content="hi")], streaming=True)
for r in res:
    logger.debug("chat resp:", r)


messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    ),
]
chatLLM(messages)

"""
## Tool Calling
ChatTongyi supports tool calling API that lets you describe tools and their arguments, and have the model return a JSON object with a tool to invoke and the inputs to that tool.

### Use with `bind_tools`
"""
logger.info("## Tool Calling")



@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


llm = ChatTongyi(model="qwen-turbo")

llm_with_tools = llm.bind_tools([multiply])

msg = llm_with_tools.invoke("What's 5 times forty two")

logger.debug(msg)

"""
### Construct args manually
"""
logger.info("### Construct args manually")


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
            "parameters": {},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或县区，比如北京市、杭州市、余杭区等。",
                    }
                },
            },
            "required": ["location"],
        },
    },
]

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the weather like in San Francisco?"),
]
chatLLM = ChatTongyi()
llm_kwargs = {"tools": tools, "result_format": "message"}
ai_message = chatLLM.bind(**llm_kwargs).invoke(messages)
ai_message

"""
## Partial Mode
Enable the large model to continue generating content from the initial text you provide.
"""
logger.info("## Partial Mode")


messages = [
    HumanMessage(
        content="""Please continue the sentence "Spring has arrived, and the earth" to express the beauty of spring and the author's joy."""
    ),
    AIMessage(
        content="Spring has arrived, and the earth", additional_kwargs={"partial": True}
    ),
]
chatLLM = ChatTongyi()
ai_message = chatLLM.invoke(messages)
ai_message

"""
## Tongyi With Vision
Qwen-VL(qwen-vl-plus/qwen-vl-max) are models that can process images.
"""
logger.info("## Tongyi With Vision")


chatLLM = ChatTongyi(model_name="qwen-vl-max")
image_message = {
    "image": "https://lilianweng.github.io/posts/2023-06-23-agent/agent-overview.png",
}
text_message = {
    "text": "summarize this picture",
}
message = HumanMessage(content=[text_message, image_message])
chatLLM.invoke([message])

"""

"""

logger.info("\n\n[DONE]", bright=True)