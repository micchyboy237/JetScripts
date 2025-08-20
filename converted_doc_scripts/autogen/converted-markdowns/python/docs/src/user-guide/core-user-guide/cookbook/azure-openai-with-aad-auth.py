from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from jet.llm.mlx.autogen_ext.mlx_chat_completion_client import AzureMLXAutogenChatLLMAdapter
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Azure MLX with AAD Auth

This guide will show you how to use the Azure MLX client with Azure Active Directory (AAD) authentication.

The identity used must be assigned the [**Cognitive Services MLX User**](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/role-based-access-control#cognitive-services-openai-user) role.

## Install Azure Identity client

The Azure identity client is used to authenticate with Azure Active Directory.
"""
logger.info("# Azure MLX with AAD Auth")

pip install azure-identity

"""
## Using the Model Client
"""
logger.info("## Using the Model Client")


token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureMLXAutogenChatLLMAdapter(
    azure_deployment="{your-azure-deployment}",
    model="{model-name, such as gpt-4o}",
    api_version="2024-02-01",
    azure_endpoint="https://{your-custom-endpoint}.openai.azure.com/",
    azure_ad_token_provider=token_provider,
)

"""

"""

See[here](https: // learn.microsoft.com/en-us/azure/ai-services/openai/how-to/managed-identity  # chat-completions) for how to use the Azure client directly or for more info.

logger.info("\n\n[DONE]", bright=True)
