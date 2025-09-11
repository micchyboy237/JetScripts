from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
import base64
import httpx
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
# How to use multimodal prompts

Here we demonstrate how to use prompt templates to format [multimodal](/docs/concepts/multimodality/) inputs to models. 

To use prompt templates in the context of multimodal data, we can templatize elements of the corresponding content block.
For example, below we define a prompt that takes a URL for an image as a parameter:
"""
logger.info("# How to use multimodal prompts")


prompt = ChatPromptTemplate(
    [
        {
            "role": "system",
            "content": "Describe the image provided.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source_type": "url",
                    "url": "{image_url}",
                },
            ],
        },
    ]
)

"""
Let's use this prompt to pass an image to a [chat model](/docs/concepts/chat_models/#multimodality):
"""
logger.info("Let's use this prompt to pass an image to a [chat model](/docs/concepts/chat_models/#multimodality):")


llm = init_chat_model("ollama:llama3.2")

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

chain = prompt | llm
response = chain.invoke({"image_url": url})
logger.debug(response.text())

"""
Note that we can templatize arbitrary elements of the content block:
"""
logger.info("Note that we can templatize arbitrary elements of the content block:")

prompt = ChatPromptTemplate(
    [
        {
            "role": "system",
            "content": "Describe the image provided.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source_type": "base64",
                    "mime_type": "{image_mime_type}",
                    "data": "{image_data}",
                    "cache_control": {"type": "{cache_type}"},
                },
            ],
        },
    ]
)



image_data = base64.b64encode(httpx.get(url).content).decode("utf-8")

chain = prompt | llm
response = chain.invoke(
    {
        "image_data": image_data,
        "image_mime_type": "image/jpeg",
        "cache_type": "ephemeral",
    }
)
logger.debug(response.text())

logger.info("\n\n[DONE]", bright=True)