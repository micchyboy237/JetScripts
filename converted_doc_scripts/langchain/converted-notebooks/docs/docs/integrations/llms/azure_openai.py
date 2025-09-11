from azure.identity import ChainedTokenCredential, ManagedIdentityCredential, AzureCliCredential
from azure.identity import DefaultAzureCredential
from jet.adapters.langchain.chat_ollama import AzureOllama
from jet.logger import logger
import ollama
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
# Azure Ollama

:::caution
You are currently on a page documenting the use of Azure Ollama [text completion models](/docs/concepts/text_llms). The latest and most popular Azure Ollama models are [chat completion models](/docs/concepts/chat_models).

Unless you are specifically using `gpt-3.5-turbo-instruct`, you are probably looking for [this page instead](/docs/integrations/chat/azure_chat_ollama/).
:::

This page goes over how to use LangChain with [Azure Ollama](https://aka.ms/azure-ollama).

The Azure Ollama API is compatible with Ollama's API.  The `ollama` Python package makes it easy to use both Ollama and Azure Ollama.  You can call Azure Ollama the same way you call Ollama with the exceptions noted below.

## API configuration
You can configure the `ollama` package to use Azure Ollama using environment variables.  The following is for `bash`:

```bash
# The API version you want to use: set this to `2023-12-01-preview` for the released version.
export OPENAI_API_VERSION=2023-12-01-preview
# The base URL for your Azure Ollama resource.  You can find this in the Azure portal under your Azure Ollama resource.
export AZURE_OPENAI_ENDPOINT=https://your-resource-name.ollama.azure.com
# The API key for your Azure Ollama resource.  You can find this in the Azure portal under your Azure Ollama resource.
# export AZURE_OPENAI_API_KEY=<your Azure Ollama API key>
```

Alternatively, you can configure the API right within your running Python environment:

```python
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
```

## Azure Active Directory Authentication
There are two ways you can authenticate to Azure Ollama:
- API Key
- Azure Active Directory (AAD)

Using the API key is the easiest way to get started. You can find your API key in the Azure portal under your Azure Ollama resource.

However, if you have complex security requirements - you may want to use Azure Active Directory. You can find more information on how to use AAD with Azure Ollama [here](https://learn.microsoft.com/en-us/azure/ai-services/ollama/how-to/managed-identity).

If you are developing locally, you will need to have the Azure CLI installed and be logged in. You can install the Azure CLI [here](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli). Then, run `az login` to log in.

Add a role an Azure role assignment `Cognitive Services Ollama User` scoped to your Azure Ollama resource. This will allow you to get a token from AAD to use with Azure Ollama. You can grant this role assignment to a user, group, service principal, or managed identity. For more information about Azure Ollama RBAC roles see [here](https://learn.microsoft.com/en-us/azure/ai-services/ollama/how-to/role-based-access-control).

# To use AAD in Python with LangChain, install the `azure-identity` package. Then, set `OPENAI_API_TYPE` to `azure_ad`. Next, use the `DefaultAzureCredential` class to get a token from AAD by calling `get_token` as shown below. Finally, set the `OPENAI_API_KEY` environment variable to the token value.

```python

# Get the Azure Credential
credential = DefaultAzureCredential()

# Set the API type to `azure_ad`
os.environ["OPENAI_API_TYPE"] = "azure_ad"
# Set the API_KEY to the token from the Azure credential
# os.environ["OPENAI_API_KEY"] = credential.get_token("https://cognitiveservices.azure.com/.default").token
```

The `DefaultAzureCredential` class is an easy way to get started with AAD authentication. You can also customize the credential chain if necessary. In the example shown below, we first try Managed Identity, then fall back to the Azure CLI. This is useful if you are running your code in Azure, but want to develop locally.

```python

credential = ChainedTokenCredential(
    ManagedIdentityCredential(),
    AzureCliCredential()
)
```

## Deployments
With Azure Ollama, you set up your own deployments of the common GPT-3 and Codex models.  When calling the API, you need to specify the deployment you want to use.

_**Note**: These docs are for the Azure text completion models. Models like GPT-4 are chat models. They have a slightly different interface, and can be accessed via the `AzureChatOllama` class. For docs on Azure chat see [Azure Chat Ollama documentation](/docs/integrations/chat/azure_chat_ollama)._

Let's say your deployment name is `gpt-35-turbo-instruct-prod`.  In the `ollama` Python API, you can specify this deployment with the `engine` parameter.  For example:

```python

client = ollama.AzureOllama(
    api_version="2023-12-01-preview",
)

response = client.completions.create(
    model="gpt-35-turbo-instruct-prod",
    prompt="Test prompt"
)
```
"""
logger.info("# Azure Ollama")

# %pip install --upgrade --quiet  langchain-ollama


os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "..."
# os.environ["AZURE_OPENAI_API_KEY"] = "..."


llm = AzureOllama(
    deployment_name="gpt-35-turbo-instruct-0914",
)

llm.invoke("Tell me a joke")

"""
We can also print the LLM and see its custom print.
"""
logger.info("We can also print the LLM and see its custom print.")

logger.debug(llm)

logger.info("\n\n[DONE]", bright=True)