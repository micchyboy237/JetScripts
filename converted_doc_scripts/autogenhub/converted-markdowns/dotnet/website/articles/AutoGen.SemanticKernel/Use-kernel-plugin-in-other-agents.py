from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
In semantic kernel, a kernel plugin is a collection of kernel functions that can be invoked during LLM calls. Semantic kernel provides a list of built-in plugins, like [core plugins](https://github.com/microsoft/semantic-kernel/tree/main/dotnet/src/Plugins/Plugins.Core), [web search plugin](https://github.com/microsoft/semantic-kernel/tree/main/dotnet/src/Plugins/Plugins.Web) and many more. You can also create your own plugins and use them in semantic kernel. Kernel plugins greatly extend the capabilities of semantic kernel and can be used to perform various tasks like web search, image search, text summarization, etc.

`AutoGen.SemanticKernel` provides a middleware called @AutoGen.SemanticKernel.KernelPluginMiddleware that allows you to use semantic kernel plugins in other AutoGen agents like @AutoGen.Ollama.OllamaChatAgent. The following example shows how to define a simple plugin with a single `GetWeather` function and use it in @AutoGen.Ollama.OllamaChatAgent.

> [!NOTE]
> You can find the complete sample code [here](https://github.com/autogenhub/autogen/blob/main/dotnet/sample/AutoGen.SemanticKernel.Sample/Use_Kernel_Functions_With_Other_Agent.cs)

### Step 1: add using statement
[!code-csharp[](../../../sample/AutoGen.SemanticKernel.Sample/Use_Kernel_Functions_With_Other_Agent.cs?name=Using)]

### Step 2: create plugin

In this step, we create a simple plugin with a single `GetWeather` function that takes a location as input and returns the weather information for that location.

[!code-csharp[](../../../sample/AutoGen.SemanticKernel.Sample/Use_Kernel_Functions_With_Other_Agent.cs?name=Create_plugin)]

### Step 3: create OllamaChatAgent and use the plugin

In this step, we firstly create a @AutoGen.SemanticKernel.KernelPluginMiddleware and register the previous plugin with it. The `KernelPluginMiddleware` will load the plugin and make the functions available for use in other agents. Followed by creating an @AutoGen.Ollama.OllamaChatAgent and register it with the `KernelPluginMiddleware`.

[!code-csharp[](../../../sample/AutoGen.SemanticKernel.Sample/Use_Kernel_Functions_With_Other_Agent.cs?name=Use_plugin)]

### Step 4: chat with OllamaChatAgent

In this final step, we start the chat with the @AutoGen.Ollama.OllamaChatAgent by asking the weather in Seattle. The `OllamaChatAgent` will use the `GetWeather` function from the plugin to get the weather information for Seattle.

[!code-csharp[](../../../sample/AutoGen.SemanticKernel.Sample/Use_Kernel_Functions_With_Other_Agent.cs?name=Send_message)]
"""
logger.info("### Step 1: add using statement")

logger.info("\n\n[DONE]", bright=True)