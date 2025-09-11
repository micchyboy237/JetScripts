from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.llms import ChatGLM
from langchain_community.llms.chatglm3 import ChatGLM3
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
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
# ChatGLM

[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) is an open bilingual language model based on General Language Model (GLM) framework, with 6.2 billion parameters. With the quantization technique, users can deploy locally on consumer-grade graphics cards (only 6GB of GPU memory is required at the INT4 quantization level). 

[ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B) is the second-generation version of the open-source bilingual (Chinese-English) chat model ChatGLM-6B. It retains the smooth conversation flow and low deployment threshold of the first-generation model, while introducing the new features like better performance, longer context and more efficient inference.

[ChatGLM3](https://github.com/THUDM/ChatGLM3) is a new generation of pre-trained dialogue models jointly released by Zhipu AI and Tsinghua KEG. ChatGLM3-6B is the open-source model in the ChatGLM3 series
"""
logger.info("# ChatGLM")

# %pip install -qU langchain langchain-community

"""
## ChatGLM3

This examples goes over how to use LangChain to interact with ChatGLM3-6B Inference for text completion.
"""
logger.info("## ChatGLM3")


template = """{question}"""
prompt = PromptTemplate.from_template(template)

endpoint_url = "http://127.0.0.1:8000/v1/chat/completions"

messages = [
    AIMessage(content="我将从美国到中国来旅游，出行前希望了解中国的城市"),
    AIMessage(content="欢迎问我任何问题。"),
]

llm = ChatGLM3(
    endpoint_url=endpoint_url,
    max_tokens=80000,
    prefix_messages=messages,
    top_p=0.9,
)

llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "北京和上海两座城市有什么不同？"

llm_chain.run(question)

"""
## ChatGLM and ChatGLM2

The following example shows how to use LangChain to interact with the ChatGLM2-6B Inference to complete text.
ChatGLM-6B and ChatGLM2-6B has the same api specs, so this example should work with both.
"""
logger.info("## ChatGLM and ChatGLM2")


template = """{question}"""
prompt = PromptTemplate.from_template(template)

endpoint_url = "http://127.0.0.1:8000"


llm = ChatGLM(
    endpoint_url=endpoint_url,
    max_token=80000,
    history=[
        ["我将从美国到中国来旅游，出行前希望了解中国的城市", "欢迎问我任何问题。"]
    ],
    top_p=0.9,
    model_kwargs={"sample_model_args": False},
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "北京和上海两座城市有什么不同？"

llm_chain.run(question)

logger.info("\n\n[DONE]", bright=True)