import asyncio
from jet.transformers.formatters import format_json
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
Before starting we need to setup some configuration, like which AI backend to use.

When using the kernel for AI requests, the kernel needs some settings like URL and
credentials to the AI models. The SDK currently supports Ollama and Azure Ollama,
other services will be added over time. If you need an Azure Ollama key, go
[here](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/quickstart?pivots=rest-api).

The following code will ask a few questions and save the settings to a local
`settings.json` configuration file, under the [config](config) folder. You can
also edit the file manually if you prefer. **Please keep the file safe.**

## Step 1

First step: choose whether you want to use the notebooks with Azure Ollama or Ollama,
setting the `useAzureOllama` boolean below.
"""
logger.info("## Step 1")

bool useAzureOllama = false;

"""
## Step 2

Run the following code. If you need to find the value and copy and paste, you can
re-run the code and continue from where you left off.
"""
logger.info("## Step 2")

async def run_async_code_b612bf5b():
    await Settings.AskAzureEndpoint(useAzureOllama);
    return 
 = asyncio.run(run_async_code_b612bf5b())
logger.success(format_json())
async def run_async_code_bc1c57e8():
    await Settings.AskModel(useAzureOllama);
    return 
 = asyncio.run(run_async_code_bc1c57e8())
logger.success(format_json())
async def run_async_code_cd77fcfb():
    await Settings.AskApiKey(useAzureOllama);
    return 
 = asyncio.run(run_async_code_cd77fcfb())
logger.success(format_json())

// Uncomment this if you're using Ollama and need to set the Org Id
async def run_async_code_0e39e81c():
    // await Settings.AskOrg(useAzureOllama);
    return 
 = asyncio.run(run_async_code_0e39e81c())
logger.success(format_json())

"""
If the code above doesn't show any error, you're good to go and run the other notebooks.

## Resetting the configuration

If you want to reset the configuration and start again, please uncomment and run the code below.
You can also edit the [config/settings.json](config/settings.json) manually if you prefer.
"""
logger.info("## Resetting the configuration")

// Uncomment this line to reset your settings and delete the file from disk.
// Settings.Reset();

"""
Now that your environment is all set up, let's dive into
[how to do basic loading of the Semantic Kernel](01-basic-loading-the-kernel.ipynb).
"""
logger.info("Now that your environment is all set up, let's dive into")

logger.info("\n\n[DONE]", bright=True)