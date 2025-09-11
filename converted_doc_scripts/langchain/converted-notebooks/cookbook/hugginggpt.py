from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_experimental.autonomous_agents import HuggingGPT
from transformers import load_tool
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
# HuggingGPT
Implementation of [HuggingGPT](https://github.com/microsoft/JARVIS). HuggingGPT is a system to connect LLMs (ChatGPT) with ML community (Hugging Face).

+ 🔥 Paper: https://arxiv.org/abs/2303.17580
+ 🚀 Project: https://github.com/microsoft/JARVIS
+ 🤗 Space: https://huggingface.co/spaces/microsoft/HuggingGPT

## Set up tools

We set up the tools available from [Transformers Agent](https://huggingface.co/docs/transformers/transformers_agents#tools). It includes a library of tools supported by Transformers and some customized tools such as image generator, video generator, text downloader and other tools.
"""
logger.info("# HuggingGPT")


hf_tools = [
    load_tool(tool_name)
    for tool_name in [
        "document-question-answering",
        "image-captioning",
        "image-question-answering",
        "image-segmentation",
        "speech-to-text",
        "summarization",
        "text-classification",
        "text-question-answering",
        "translation",
        "huggingface-tools/text-to-image",
        "huggingface-tools/text-to-video",
        "text-to-speech",
        "huggingface-tools/text-download",
        "huggingface-tools/image-transformation",
    ]
]

"""
## Setup model and HuggingGPT

We create an instance of HuggingGPT and use ChatGPT as the controller to rule the above tools.
"""
logger.info("## Setup model and HuggingGPT")


llm = ChatOllama(model_name="gpt-3.5-turbo")
agent = HuggingGPT(llm, hf_tools)

"""
## Run an example

Given a text, show a related image and video.
"""
logger.info("## Run an example")

agent.run("please show me a video and an image of 'a boy is running'")

logger.info("\n\n[DONE]", bright=True)
