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
# Basic Loading of the Kernel

The Semantic Kernel SDK can be imported from the following nuget feed:
"""
logger.info("# Basic Loading of the Kernel")



"""
After adding the nuget package, you can instantiate the kernel:
"""
logger.info("After adding the nuget package, you can instantiate the kernel:")

using Microsoft.SemanticKernel;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.DependencyInjection;
using Kernel = Microsoft.SemanticKernel.Kernel;

// Inject your logger
// see Microsoft.Extensions.Logging.ILogger @ https://learn.microsoft.com/dotnet/core/extensions/logging
ILoggerFactory myLoggerFactory = NullLoggerFactory.Instance;

var builder = Kernel.CreateBuilder();
builder.Services.AddSingleton(myLoggerFactory);

var kernel = builder.Build();

"""
When using the kernel for AI requests, the kernel needs some settings like URL and credentials to the AI models.

The SDK currently supports Ollama, Azure Ollama and HuggingFace. It's also possible to create your own connector and use AI provider of your choice.

If you need an Azure Ollama key, go [here](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/quickstart?pivots=rest-api).
"""
logger.info("When using the kernel for AI requests, the kernel needs some settings like URL and credentials to the AI models.")

Kernel.CreateBuilder()
.AddAzureOllamaChatCompletion(
    "my-finetuned-model",                   // Azure Ollama *Deployment Name*
    "https://contoso.openai.azure.com/",    // Azure Ollama *Endpoint*
    "...your Azure Ollama Key...",          // Azure Ollama *Key*
    serviceId: "Azure_curie"                // alias used in the prompt templates' config.json
)
.AddOllamaChatCompletion(
    "llama3.2",                        // Ollama Model Name
    "...your Ollama API Key...",            // Ollama API key
    "...your Ollama Org ID...",             // *optional* Ollama Organization ID
    serviceId: "Ollama_davinci"             // alias used in the prompt templates' config.json
);

"""
When working with multiple backends and multiple models, the **first backend** defined
is also the "**default**" used in these scenarios:

* a prompt configuration doesn't specify which AI backend to use
* a prompt configuration requires a backend unknown to the kernel

Great, now that you're familiar with setting up the Semantic Kernel, let's see [how we can use it to run prompts](02-running-prompts-from-file.ipynb).
"""
logger.info("When working with multiple backends and multiple models, the **first backend** defined")

logger.info("\n\n[DONE]", bright=True)