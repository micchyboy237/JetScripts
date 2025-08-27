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
# Creating a basic chat experience with kernel arguments

In this example, we show how you can build a simple chat bot by sending and updating arguments with your requests. 

We introduce the Kernel Arguments object which in this demo functions similarly as a key-value store that you can use when running the kernel.  

In this chat scenario, as the user talks back and forth with the bot, the arguments get populated with the history of the conversation. During each new run of the kernel, the arguments will be provided to the AI with content.
"""
logger.info("# Creating a basic chat experience with kernel arguments")

using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.Ollama;
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
Let's define a prompt outlining a dialogue chat bot.
"""
logger.info("Let's define a prompt outlining a dialogue chat bot.")

const string skPrompt = @"
ChatBot can have a conversation with you about any topic.
It can give explicit instructions or say 'I don't know' if it does not have an answer.

{{$history}}
User: {{$userInput}}
ChatBot:";

var executionSettings = new OllamaPromptExecutionSettings
{
    MaxTokens = 2000,
    Temperature = 0.7,
    TopP = 0.5
};

"""
Register your semantic function
"""
logger.info("Register your semantic function")

var chatFunction = kernel.CreateFunctionFromPrompt(skPrompt, executionSettings);

"""
Initialize your arguments
"""
logger.info("Initialize your arguments")

var history = "";
var arguments = new KernelArguments()
{
    ["history"] = history
};

"""
Chat with the Bot
"""
logger.info("Chat with the Bot")

var userInput = "Hi, I'm looking for book suggestions";
arguments["userInput"] = userInput;

async def run_async_code_691f9e68():
    async def run_async_code_1a5644fe():
        var bot_answer = await chatFunction.InvokeAsync(kernel, arguments);
        return var bot_answer
    var bot_answer = asyncio.run(run_async_code_1a5644fe())
    logger.success(format_json(var bot_answer))
    return var bot_answer
var bot_answer = asyncio.run(run_async_code_691f9e68())
logger.success(format_json(var bot_answer))

"""
Update the history with the output and set this as the new input value for the next request
"""
logger.info("Update the history with the output and set this as the new input value for the next request")

history += $"\nUser: {userInput}\nAI: {bot_answer}\n";
arguments["history"] = history;

Console.WriteLine(history);

"""
Keep Chatting!
"""
logger.info("Keep Chatting!")

Func<string, Task> Chat = async (string input) => {
    // Save new message in the arguments
    arguments["userInput"] = input;

    // Process the user message and get an answer
    async def run_async_code_831abfd1():
        async def run_async_code_2e41f66a():
            var answer = await chatFunction.InvokeAsync(kernel, arguments);
            return var answer
        var answer = asyncio.run(run_async_code_2e41f66a())
        logger.success(format_json(var answer))
        return var answer
    var answer = asyncio.run(run_async_code_831abfd1())
    logger.success(format_json(var answer))

    // Append the new interaction to the chat history
    var result = $"\nUser: {input}\nAI: {answer}\n";
    history += result;

    arguments["history"] = history;

    // Show the response
    Console.WriteLine(result);
};

async def run_async_code_46a2df71():
    await Chat("I would like a non-fiction book suggestion about Greece history. Please only list one book.");
    return 
 = asyncio.run(run_async_code_46a2df71())
logger.success(format_json())

async def run_async_code_04de32b8():
    await Chat("that sounds interesting, what are some of the topics I will learn about?");
    return 
 = asyncio.run(run_async_code_04de32b8())
logger.success(format_json())

async def run_async_code_ccb6ac28():
    await Chat("Which topic from the ones you listed do you think most people find interesting?");
    return 
 = asyncio.run(run_async_code_ccb6ac28())
logger.success(format_json())

async def run_async_code_04e588a0():
    await Chat("could you list some more books I could read about the topic(s) you mentioned?");
    return 
 = asyncio.run(run_async_code_04e588a0())
logger.success(format_json())

"""
After chatting for a while, we have built a growing history, which we are attaching to each prompt and which contains the full conversation. Let's take a look!
"""
logger.info("After chatting for a while, we have built a growing history, which we are attaching to each prompt and which contains the full conversation. Let's take a look!")

Console.WriteLine(history);

logger.info("\n\n[DONE]", bright=True)