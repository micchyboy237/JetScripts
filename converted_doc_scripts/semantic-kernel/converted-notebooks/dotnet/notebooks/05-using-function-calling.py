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
# Introduction to the Function Calling

The most powerful feature of chat completion is the ability to call functions from the model. This allows you to create a chat bot that can interact with your existing code, making it possible to automate business processes, create code snippets, and more.

With Semantic Kernel, we simplify the process of using function calling by automatically describing your functions and their parameters to the model and then handling the back-and-forth communication between the model and your code.

Read more about it [here](https://learn.microsoft.com/en-us/semantic-kernel/concepts/ai-services/chat-completion/function-calling).
"""
logger.info("# Introduction to the Function Calling")

using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.Ollama;
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
### Setting Up Execution Settings

Using `FunctionChoiceBehavior.Auto()` will enable automatic function calling. There are also other options like `Required` or `None` which allow to control function calling behavior. More information about it can be found [here](https://learn.microsoft.com/en-gb/semantic-kernel/concepts/ai-services/chat-completion/function-calling/function-choice-behaviors?pivots=programming-language-csharp).
"""
logger.info("### Setting Up Execution Settings")

OllamaPromptExecutionSettings openAIPromptExecutionSettings = new()
{
    FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()
};

"""
### Providing plugins to the Kernel
Function calling needs an information about available plugins/functions. Here we'll import the `SummarizePlugin` and `WriterPlugin` we have defined on disk.
"""
logger.info("### Providing plugins to the Kernel")

var pluginsDirectory = Path.Combine(System.IO.Directory.GetCurrentDirectory(), "..", "..", "prompt_template_samples");

kernel.ImportPluginFromPromptDirectory(Path.Combine(pluginsDirectory, "SummarizePlugin"));
kernel.ImportPluginFromPromptDirectory(Path.Combine(pluginsDirectory, "WriterPlugin"));

"""
Define your ASK. What do you want the Kernel to do?
"""
logger.info("Define your ASK. What do you want the Kernel to do?")

var ask = "Tomorrow is Valentine's day. I need to come up with a few date ideas. My significant other likes poems so write them in the form of a poem.";

"""
Since we imported available plugins to Kernel and defined the ask, we can now invoke a prompt with all the provided information. 

We can run function calling with Kernel, if we are interested in result only.
"""
logger.info("Since we imported available plugins to Kernel and defined the ask, we can now invoke a prompt with all the provided information.")

async def run_async_code_fe9f7b07():
    async def run_async_code_118daa71():
        var result = await kernel.InvokePromptAsync(ask, new(openAIPromptExecutionSettings));
        return var result
    var result = asyncio.run(run_async_code_118daa71())
    logger.success(format_json(var result))
    return var result
var result = asyncio.run(run_async_code_fe9f7b07())
logger.success(format_json(var result))

Console.WriteLine(result);

"""
But we can also run it with `IChatCompletionService` to have an access to `ChatHistory` object, which allows us to see which functions were called as part of a function calling process. Note that passing a Kernel as a parameter to `GetChatMessageContentAsync` method is required, since Kernel holds an information about available plugins.
"""
logger.info("But we can also run it with `IChatCompletionService` to have an access to `ChatHistory` object, which allows us to see which functions were called as part of a function calling process. Note that passing a Kernel as a parameter to `GetChatMessageContentAsync` method is required, since Kernel holds an information about available plugins.")

using Microsoft.SemanticKernel.ChatCompletion;

var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();

var chatHistory = new ChatHistory();

chatHistory.AddUserMessage(ask);

async def run_async_code_9948af67():
    async def run_async_code_ebc257b3():
        var chatCompletionResult = await chatCompletionService.GetChatMessageContentAsync(chatHistory, openAIPromptExecutionSettings, kernel);
        return var chatCompletionResult
    var chatCompletionResult = asyncio.run(run_async_code_ebc257b3())
    logger.success(format_json(var chatCompletionResult))
    return var chatCompletionResult
var chatCompletionResult = asyncio.run(run_async_code_9948af67())
logger.success(format_json(var chatCompletionResult))

Console.WriteLine($"Result: {chatCompletionResult}\n");
Console.WriteLine($"Chat history: {JsonSerializer.Serialize(chatHistory)}\n");

logger.info("\n\n[DONE]", bright=True)