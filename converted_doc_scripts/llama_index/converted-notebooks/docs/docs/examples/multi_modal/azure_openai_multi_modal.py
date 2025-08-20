from IPython.display import HTML
from jet.logger import CustomLogger
from llama_index.core.llms import (
ChatMessage,
ImageBlock,
TextBlock,
MessageRole,
)
from llama_index.core.schema import Document, MediaResource
from llama_index.llms.azure_openai import AzureMLX
import base64
import os
import requests
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/azure_openai_multi_modal.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Multi-Modal LLM using Azure MLX GPT-4o mini for image reasoning

In this notebook, we show how to use GPT-4o mini with the **Azure** MLX LLM class/abstraction for image understanding/reasoning. For a more complete example, please visit [this notebook](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/multi_modal/openai_multi_modal.ipynb).
"""
logger.info("# Multi-Modal LLM using Azure MLX GPT-4o mini for image reasoning")

# %pip install llama-index-llms-azure-openai

"""
## Prerequisites

1. Setup an Azure subscription - you can create one for free [here](https://azure.microsoft.com/en-us/free/cognitive-services/)
2. Apply for access to Azure MLX Service [here](https://customervoice.microsoft.com/Pages/ResponsePage.aspx?id=v4j5cvGGr0GRqy180BHbR7en2Ais5pxKtso_Pz4b1_xUOFA5Qk1UWDRBMjg0WFhPMkIzTzhKQ1dWNyQlQCN0PWcu) 
3. Create a resource in the Azure portal [here](https://portal.azure.com/?microsoft_azure_marketplace_ItemHideKey=microsoft_openai_tip#create/Microsoft.CognitiveServicesMLX)
4. Deploy a model in Azure MLX Studio [here](https://oai.azure.com/)


You can find more details in [this guide.](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal)

Note down the **"model name"** and **"deployment name"**, you'll need it when connecting to your LLM.

##  Use GPT-4o mini to understand Images from URLs / base64
"""
logger.info("## Prerequisites")


# os.environ["AZURE_OPENAI_API_KEY"] = "xxx"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://YOUR_URL.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2024-02-15-preview"

"""
## Initialize `AzureMLX` and Load Images from URLs

Unlike regular `MLX`, you need to pass the `engine` argument in addition to `model`. The `engine` is the name you 
gave to your model when you deployed it in Azure MLX Studio.
"""
logger.info("## Initialize `AzureMLX` and Load Images from URLs")


azure_openai_llm = AzureMLX(
    engine="my-qwen3-1.7b-4bit-mini",
    model="qwen3-1.7b-4bit-mini",
    max_new_tokens=300,
)

"""
Alternatively, you can also skip setting environment variables, and pass the parameters in directly via constructor.
"""
logger.info("Alternatively, you can also skip setting environment variables, and pass the parameters in directly via constructor.")

azure_openai_llm = AzureMLX(
    azure_endpoint="https://YOUR_URL.openai.azure.com/",
    engine="my-qwen3-1.7b-4bit-mini",
    api_version="2024-02-15-preview",
    model="qwen3-1.7b-4bit-mini",
    max_new_tokens=300,
    api_key="xxx",
    supports_content_blocks=True,
)


image_url = "https://www.visualcapitalist.com/wp-content/uploads/2023/10/US_Mortgage_Rate_Surge-Sept-11-1.jpg"

response = requests.get(image_url)
if response.status_code != 200:
    raise ValueError("Error: Could not retrieve image from URL.")
img_data = base64.b64encode(response.content)

image_document = Document(image_resource=MediaResource(data=img_data))


src = f'<img width=400 src="data:{image_document.image_resource.mimetype};base64,{image_document.image_resource.data.decode("utf-8")}"/>'
HTML(src)

"""
### Complete a prompt with an image
"""
logger.info("### Complete a prompt with an image")


msg = ChatMessage(
    role=MessageRole.USER,
    blocks=[
        TextBlock(text="Describe the images as an alternative text"),
        ImageBlock(image=image_document.image_resource.data),
    ],
)

response = azure_openai_llm.chat(messages=[msg])

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)