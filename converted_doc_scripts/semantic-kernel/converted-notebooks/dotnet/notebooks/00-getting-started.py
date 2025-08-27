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
#### Watch the Getting Started Quick Start [Video](https://aka.ms/SK-Getting-Started-Notebook)

> [!IMPORTANT]
> You will need an [.NET 8 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/8.0) and [Polyglot](https://marketplace.visualstudio.com/items?itemName=ms-dotnettools.dotnet-interactive-vscode) to get started with this notebook using .NET Interactive.

**Step 1**: Configure your AI service credentials

Use [this notebook](0-AI-settings.ipynb) first, to choose whether to run these notebooks with Ollama or Azure Ollama,
and to save your credentials in the configuration file.
"""
logger.info("#### Watch the Getting Started Quick Start [Video](https://aka.ms/SK-Getting-Started-Notebook)")

// Load some helper functions, e.g. to load values from settings.json

"""
**Step 2**: Import Semantic Kernel SDK from NuGet
"""

// Import Semantic Kernel

"""
**Step 3**: Instantiate the Kernel
"""

using Microsoft.SemanticKernel;
using Kernel = Microsoft.SemanticKernel.Kernel;

//Create Kernel builder
var builder = Kernel.CreateBuilder();

// Configure AI service credentials used by the kernel
var (useAzureOllama, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();

if (useAzureOllama)
    builder.AddAzureOllamaChatCompletion(model, azureEndpoint, apiKey);
else
    builder.AddOllamaChatCompletion(model, apiKey, orgId);

var kernel = builder.Build();

"""
**Step 4**: Load and Run a Plugin
"""

// FunPlugin directory path
var funPluginDirectoryPath = Path.Combine(System.IO.Directory.GetCurrentDirectory(), "..", "..", "prompt_template_samples", "FunPlugin");

// Load the FunPlugin from the Plugins Directory
var funPluginFunctions = kernel.ImportPluginFromPromptDirectory(funPluginDirectoryPath);

// Construct arguments
var arguments = new KernelArguments() { ["input"] = "time travel to dinosaur age" };

// Run the Function called Joke
async def run_async_code_aed22e96():
    async def run_async_code_35eeccde():
        var result = await kernel.InvokeAsync(funPluginFunctions["Joke"], arguments);
        return var result
    var result = asyncio.run(run_async_code_35eeccde())
    logger.success(format_json(var result))
    return var result
var result = asyncio.run(run_async_code_aed22e96())
logger.success(format_json(var result))

// Return the result to the Notebook
Console.WriteLine(result);

logger.info("\n\n[DONE]", bright=True)