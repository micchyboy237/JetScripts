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
# Running Semantic Functions Inline

The [previous notebook](./02-running-prompts-from-file.ipynb)
showed how to define a semantic function using a prompt template stored on a file.

In this notebook, we'll show how to use the Semantic Kernel to define functions inline with your C# code. This can be useful in a few scenarios:

* Dynamically generating the prompt using complex rules at runtime
* Writing prompts by editing C# code instead of TXT files. 
* Easily creating demos, like this document

Prompt templates are defined using the SK template language, which allows to reference variables and functions. Read [this doc](https://aka.ms/sk/howto/configurefunction) to learn more about the design decisions for prompt templating. 

For now we'll use only the `{{$input}}` variable, and see more complex templates later.

Almost all semantic function prompts have a reference to `{{$input}}`, which is the default way
a user can import content from the kernel arguments.

Prepare a semantic kernel instance first, loading also the AI backend settings defined in the [Setup notebook](0-AI-settings.ipynb):
"""
logger.info("# Running Semantic Functions Inline")

using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.Ollama;
using Microsoft.SemanticKernel.TemplateEngine;
using Kernel = Microsoft.SemanticKernel.Kernel;

var builder = Kernel.CreateBuilder();

// Configure AI service credentials used by the kernel
var (useAzureOllama, model, azureEndpoint, apiKey, orgId) = Settings.LoadFromFile();

if (useAzureOllama)
    builder.AddAzureOllamaChatCompletion(model, azureEndpoint, apiKey);
else
    builder.AddOllamaChatCompletion(model, apiKey, orgId);

var kernel = builder.Build();

"""
Let's create a semantic function used to summarize content:
"""
logger.info("Let's create a semantic function used to summarize content:")

string skPrompt = """
{{$input}}

Summarize the content above.
""";

"""
Let's configure the prompt, e.g. allowing for some creativity and a sufficient number of tokens.
"""
logger.info("Let's configure the prompt, e.g. allowing for some creativity and a sufficient number of tokens.")

var executionSettings = new OllamaPromptExecutionSettings
{
    MaxTokens = 2000,
    Temperature = 0.2,
    TopP = 0.5
};

"""
The following code prepares an instance of the template, passing in the TXT and configuration above, 
and a couple of other parameters (how to render the TXT and how the template can access other functions).

This allows to see the prompt before it's sent to AI.
"""
logger.info("The following code prepares an instance of the template, passing in the TXT and configuration above,")

var promptTemplateConfig = new PromptTemplateConfig(skPrompt);

var promptTemplateFactory = new KernelPromptTemplateFactory();
var promptTemplate = promptTemplateFactory.Create(promptTemplateConfig);

async def run_async_code_1d1a45f6():
    async def run_async_code_3ba396f6():
        var renderedPrompt = await promptTemplate.RenderAsync(kernel);
        return var renderedPrompt
    var renderedPrompt = asyncio.run(run_async_code_3ba396f6())
    logger.success(format_json(var renderedPrompt))
    return var renderedPrompt
var renderedPrompt = asyncio.run(run_async_code_1d1a45f6())
logger.success(format_json(var renderedPrompt))

Console.WriteLine(renderedPrompt);

"""
Let's transform the prompt template into a function that the kernel can execute:
"""
logger.info("Let's transform the prompt template into a function that the kernel can execute:")

var summaryFunction = kernel.CreateFunctionFromPrompt(skPrompt, executionSettings);

"""
Set up some content to summarize, here's an extract about Demo, an ancient Greek poet, taken from [Wikipedia](https://en.wikipedia.org/wiki/Demo_(ancient_Greek_poet)).
"""
logger.info("Set up some content to summarize, here's an extract about Demo, an ancient Greek poet, taken from [Wikipedia](https://en.wikipedia.org/wiki/Demo_(ancient_Greek_poet)).")

var input = """
Demo (ancient Greek poet)
From Wikipedia, the free encyclopedia
Demo or Damo (Greek: Δεμώ, Δαμώ; fl. c. AD 200) was a Greek woman of the Roman period, known for a single epigram, engraved upon the Colossus of Memnon, which bears her name. She speaks of herself therein as a lyric poetess dedicated to the Muses, but nothing is known of her life.[1]
Identity
Demo was evidently Greek, as her name, a traditional epithet of Demeter, signifies. The name was relatively common in the Hellenistic world, in Egypt and elsewhere, and she cannot be further identified. The date of her visit to the Colossus of Memnon cannot be established with certainty, but internal evidence on the left leg suggests her poem was inscribed there at some point in or after AD 196.[2]
Epigram
There are a number of graffiti inscriptions on the Colossus of Memnon. Following three epigrams by Julia Balbilla, a fourth epigram, in elegiac couplets, entitled and presumably authored by "Demo" or "Damo" (the Greek inscription is difficult to read), is a dedication to the Muses.[2] The poem is traditionally published with the works of Balbilla, though the internal evidence suggests a different author.[1]
In the poem, Demo explains that Memnon has shown her special respect. In return, Demo offers the gift for poetry, as a gift to the hero. At the end of this epigram, she addresses Memnon, highlighting his divine status by recalling his strength and holiness.[2]
Demo, like Julia Balbilla, writes in the artificial and poetic Aeolic dialect. The language indicates she was knowledgeable in Homeric poetry—'bearing a pleasant gift', for example, alludes to the use of that phrase throughout the Iliad and Odyssey.[a][2]
""";

"""
...and run the summary function:
"""

async def run_async_code_3de957d9():
    async def run_async_code_9b49e4d3():
        var summaryResult = await kernel.InvokeAsync(summaryFunction, new() { ["input"] = input });
        return var summaryResult
    var summaryResult = asyncio.run(run_async_code_9b49e4d3())
    logger.success(format_json(var summaryResult))
    return var summaryResult
var summaryResult = asyncio.run(run_async_code_3de957d9())
logger.success(format_json(var summaryResult))

Console.WriteLine(summaryResult);

"""
The code above shows all the steps, to understand how the function is composed step by step. However, the kernel
includes also some helpers to achieve the same more concisely.

The same function above can be executed with less code:
"""
logger.info("The code above shows all the steps, to understand how the function is composed step by step. However, the kernel")

string skPrompt = """
{{$input}}

Summarize the content above.
""";

async def run_async_code_bc04a437():
    async def run_async_code_2280e553():
        var result = await kernel.InvokePromptAsync(skPrompt, new() { ["input"] = input });
        return var result
    var result = asyncio.run(run_async_code_2280e553())
    logger.success(format_json(var result))
    return var result
var result = asyncio.run(run_async_code_bc04a437())
logger.success(format_json(var result))

Console.WriteLine(result);

"""
Here's one more example of how to write an inline Semantic Function that gives a TLDR for a piece of text.
"""
logger.info("Here's one more example of how to write an inline Semantic Function that gives a TLDR for a piece of text.")

string skPrompt = @"
{{$input}}

Give me the TLDR in 5 words.
";

var textToSummarize = @"
    1) A robot may not injure a human being or, through inaction,
    allow a human being to come to harm.

    2) A robot must obey orders given it by human beings except where
    such orders would conflict with the First Law.

    3) A robot must protect its own existence as long as such protection
    does not conflict with the First or Second Law.
";

async def run_async_code_b501c28d():
    var result = await kernel.InvokePromptAsync(skPrompt, new() { ["input"] = textToSummarize });
    return var result
var result = asyncio.run(run_async_code_b501c28d())
logger.success(format_json(var result))

Console.WriteLine(result);

logger.info("\n\n[DONE]", bright=True)