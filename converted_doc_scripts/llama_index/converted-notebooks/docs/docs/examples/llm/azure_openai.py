from IPython.display import Image
from jet.logger import CustomLogger
from llama_index.core.llms import ChatMessage
from llama_index.llms.azure_openai import AzureMLX
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/azure_openai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Azure MLX

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Azure MLX")

# %pip install llama-index-llms-azure-openai

# !pip install llama-index

"""
## Prerequisites

1. Setup an Azure subscription - you can create one for free [here](https://azure.microsoft.com/en-us/free/cognitive-services/)
2. Apply for access to Azure MLX Service [here](https://customervoice.microsoft.com/Pages/ResponsePage.aspx?id=v4j5cvGGr0GRqy180BHbR7en2Ais5pxKtso_Pz4b1_xUOFA5Qk1UWDRBMjg0WFhPMkIzTzhKQ1dWNyQlQCN0PWcu) 
3. Create a resource in the Azure portal [here](https://portal.azure.com/?microsoft_azure_marketplace_ItemHideKey=microsoft_openai_tip#create/Microsoft.CognitiveServicesMLX)
4. Deploy a model in Azure MLX Studio [here](https://oai.azure.com/)


You can find more details in [this guide.](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal)

Note down the **"model name"** and **"deployment name"**, you'll need it when connecting to your LLM.

## Environment Setup

### Find your setup information - API base, API key, deployment name (i.e. engine), etc

To find the setup information necessary, do the following setups:
1. Go to the Azure MLX Studio [here](https://oai.azure.com/)
2. Go to the chat or completions playground (depending on which LLM you are setting up)
3. Click "view code" (shown in image below)
"""
logger.info("## Prerequisites")


Image(filename="./azure_playground.png")

"""
4. Note down the `api_type`, `api_base`, `api_version`, `engine` (this should be the same as the "deployment name" from before), and the `key`
"""
logger.info("4. Note down the `api_type`, `api_base`, `api_version`, `engine` (this should be the same as the "deployment name" from before), and the `key`")


Image(filename="./azure_env.png")

"""
### Configure environment variables

Using Azure deployment of MLX models is very similar to normal MLX. 
You just need to configure a couple more environment variables.

- `OPENAI_API_VERSION`: set this to `2023-07-01-preview`
    This may change in the future.
- `AZURE_OPENAI_ENDPOINT`: your endpoint should look like the following
    https://YOUR_RESOURCE_NAME.openai.azure.com/
# - `AZURE_OPENAI_API_KEY`: your API key
"""
logger.info("### Configure environment variables")


# os.environ["AZURE_OPENAI_API_KEY"] = "<your-api-key>"
os.environ[
    "AZURE_OPENAI_ENDPOINT"
] = "https://<your-resource-name>.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"

"""
## Use your LLM
"""
logger.info("## Use your LLM")


"""
Unlike normal `MLX`, you need to pass a `engine` argument in addition to `model`. The `engine` is the name of your model deployment you selected in Azure MLX Studio. See previous section on "find your setup information" for more details.
"""
logger.info("Unlike normal `MLX`, you need to pass a `engine` argument in addition to `model`. The `engine` is the name of your model deployment you selected in Azure MLX Studio. See previous section on "find your setup information" for more details.")

llm = AzureMLX(
    engine="simon-llm", model="gpt-35-turbo-16k", temperature=0.0
)

"""
Alternatively, you can also skip setting environment variables, and pass the parameters in directly via constructor.
"""
logger.info("Alternatively, you can also skip setting environment variables, and pass the parameters in directly via constructor.")

llm = AzureMLX(
    engine="my-custom-llm",
    model="gpt-35-turbo-16k",
    temperature=0.0,
    azure_endpoint="https://<your-resource-name>.openai.azure.com/",
    api_key="<your-api-key>",
    api_version="2023-07-01-preview",
)

"""
Use the `complete` endpoint for text completion
"""
logger.info("Use the `complete` endpoint for text completion")

response = llm.complete("The sky is a beautiful blue and")
logger.debug(response)

response = llm.stream_complete("The sky is a beautiful blue and")
for r in response:
    logger.debug(r.delta, end="")

"""
Use the `chat` endpoint for conversation
"""
logger.info("Use the `chat` endpoint for conversation")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with colorful personality."
    ),
    ChatMessage(role="user", content="Hello"),
]

response = llm.chat(messages)
logger.debug(response)

response = llm.stream_chat(messages)
for r in response:
    logger.debug(r.delta, end="")

"""
Rather than adding same parameters to each chat or completion call, you can set them at a per-instance level with `additional_kwargs`.
"""
logger.info("Rather than adding same parameters to each chat or completion call, you can set them at a per-instance level with `additional_kwargs`.")

llm = AzureMLX(
    engine="simon-llm",
    model="gpt-35-turbo-16k",
    temperature=0.0,
    additional_kwargs={"user": "your_user_id"},
)

logger.info("\n\n[DONE]", bright=True)