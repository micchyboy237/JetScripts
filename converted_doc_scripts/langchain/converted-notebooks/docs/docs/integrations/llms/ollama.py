from IPython.display import HTML, display
from PIL import Image
from io import BytesIO
from jet.adapters.langchain.chat_ollama import OllamaLLM
from jet.adapters.langchain.chat_ollama.llms import OllamaLLM
from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate
import base64
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
sidebar_label: Ollama
---

# OllamaLLM

:::caution
You are currently on a page documenting the use of Ollama models as [text completion models](/docs/concepts/text_llms). Many popular Ollama models are [chat completion models](/docs/concepts/chat_models).

You may be looking for [this page instead](/docs/integrations/chat/ollama/).
:::

This page goes over how to use LangChain to interact with `Ollama` models.

## Installation
"""
logger.info("# OllamaLLM")

# %pip install -U langchain-ollama

"""
## Setup

First, follow [these instructions](https://github.com/ollama/ollama?tab=readme-ov-file#ollama) to set up and run a local Ollama instance:

* [Download](https://ollama.ai/download) and install Ollama onto the available supported platforms (including Windows Subsystem for Linux aka WSL, macOS, and Linux)
    * macOS users can install via Homebrew with `brew install ollama` and start with `brew services start ollama`
* Fetch available LLM model via `ollama pull <name-of-model>`
    * View a list of available models via the [model library](https://ollama.ai/library)
    * e.g., `ollama pull llama3`
* This will download the default tagged version of the model. Typically, the default points to the latest, smallest sized-parameter model.

> On Mac, the models will be download to `~/.ollama/models`
>
> On Linux (or WSL), the models will be stored at `/usr/share/ollama/.ollama/models`

* Specify the exact version of the model of interest as such `ollama pull vicuna:13b-v1.5-16k-q4_0` (View the [various tags for the `Vicuna`](https://ollama.ai/library/vicuna/tags) model in this instance)
* To view all pulled models, use `ollama list`
* To chat directly with a model from the command line, use `ollama run <name-of-model>`
* View the [Ollama documentation](https://github.com/ollama/ollama/tree/main/docs) for more commands. You can run `ollama help` in the terminal to see available commands.

## Usage
"""
logger.info("## Setup")


template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="llama3.1")

chain = prompt | model

chain.invoke({"question": "What is LangChain?"})

"""
## Multi-modal

Ollama has support for multi-modal LLMs, such as [bakllava](https://ollama.com/library/bakllava) and [llava](https://ollama.com/library/llava).

    ollama pull bakllava

Be sure to update Ollama so that you have the most recent version to support multi-modal.
"""
logger.info("## Multi-modal")

# %pip install pillow




def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def plt_img_base64(img_base64):
    """
    Display base64 encoded string as image

    :param img_base64:  Base64 string
    """
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    display(HTML(image_html))


file_path = "../../../static/img/ollama_example_img.jpg"
pil_image = Image.open(file_path)
image_b64 = convert_to_base64(pil_image)
plt_img_base64(image_b64)


llm = OllamaLLM(model="bakllava")

llm_with_image_context = llm.bind(images=[image_b64])
llm_with_image_context.invoke("What is the dollar based gross retention rate:")

"""
## API reference

For detailed documentation of all ChatOllama features and configurations head to the [API reference](https://python.langchain.com/api_reference/ollama/llms/jet.adapters.langchain.chat_ollama.llms.OllamaLLM.html).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)