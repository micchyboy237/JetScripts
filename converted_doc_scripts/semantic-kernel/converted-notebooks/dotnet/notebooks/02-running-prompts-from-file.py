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
# How to run a semantic plugins from file
Now that you're familiar with Kernel basics, let's see how the kernel allows you to run Semantic Plugins and Semantic Functions stored on disk. 

A Semantic Plugin is a collection of Semantic Functions, where each function is defined with natural language that can be provided with a text file. 

Refer to our [glossary](../../docs/GLOSSARY.md) for an in-depth guide to the terms.

The repository includes some examples under the [samples](https://github.com/microsoft/semantic-kernel/tree/main/samples) folder.

For instance, [this](../../samples/plugins/FunPlugin/Joke/skprompt.txt) is the **Joke function** part of the **FunPlugin plugin**:

```
WRITE EXACTLY ONE JOKE or HUMOROUS STORY ABOUT THE TOPIC BELOW.
JOKE MUST BE:
- G RATED
- WORKPLACE/FAMILY SAFE
NO SEXISM, RACISM OR OTHER BIAS/BIGOTRY.
BE CREATIVE AND FUNNY. I WANT TO LAUGH.
+++++
{{$input}}
+++++
```

Note the special **`{{$input}}`** token, which is a variable that is automatically passed when invoking the function, commonly referred to as a "function parameter". 

We'll explore later how functions can accept multiple variables, as well as invoke other functions.

In the same folder you'll notice a second [config.json](../../samples/plugins/FunPlugin/Joke/config.json) file. The file is optional, and is used to set some parameters for large language models like Temperature, TopP, Stop Sequences, etc.

```
{
  "schema": 1,
  "description": "Generate a funny joke",
  "execution_settings": [
    {
      "max_tokens": 1000,
      "temperature": 0.9,
      "top_p": 0.0,
      "presence_penalty": 0.0,
      "frequency_penalty": 0.0
    }
  ]
}
```

Given a semantic function defined by these files, this is how to load and use a file based semantic function.

Configure and create the kernel, as usual, loading also the AI backend settings defined in the [Setup notebook](0-AI-settings.ipynb):
"""
logger.info("# How to run a semantic plugins from file")

using Microsoft.SemanticKernel;
using Kernel = Microsoft.SemanticKernel.Kernel;

var builder = Kernel.CreateBuilder();

// Configure AI backend used by the kernel
var (useAzureOllama, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();

if (useAzureOllama)
    builder.AddAzureOllamaChatCompletion(model, azureEndpoint, apiKey);
else
    builder.AddOllamaChatCompletion(model, apiKey, orgId);

var kernel = builder.Build();

"""
Import the plugin and all its functions:
"""
logger.info("Import the plugin and all its functions:")

// FunPlugin directory path
var funPluginDirectoryPath = Path.Combine(System.IO.Directory.GetCurrentDirectory(), "..", "..", "prompt_template_samples", "FunPlugin");

// Load the FunPlugin from the Plugins Directory
var funPluginFunctions = kernel.ImportPluginFromPromptDirectory(funPluginDirectoryPath);

"""
How to use the plugin functions, e.g. generate a joke about "*time travel to dinosaur age*":
"""
logger.info("How to use the plugin functions, e.g. generate a joke about "*time travel to dinosaur age*":")

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

"""
Great, now that you know how to load a plugin from disk, let's show how you can [create and run a semantic function inline.](./03-semantic-function-inline.ipynb)
"""
logger.info("Great, now that you know how to load a plugin from disk, let's show how you can [create and run a semantic function inline.](./03-semantic-function-inline.ipynb)")

logger.info("\n\n[DONE]", bright=True)