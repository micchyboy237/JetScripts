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
The following example shows how to create a `GetWeatherAsync` function and pass it to @AutoGen.Ollama.OllamaChatAgent.

Firstly, you need to install the following packages:
"""
logger.info("The following example shows how to create a `GetWeatherAsync` function and pass it to @AutoGen.Ollama.OllamaChatAgent.")

<ItemGroup>
    <PackageReference Include="AutoGen.Ollama" Version="AUTOGEN_VERSION" />
    <PackageReference Include="AutoGen.SourceGenerator" Version="AUTOGEN_VERSION" />
</ItemGroup>

"""
> [!Note]
> The `AutoGen.SourceGenerator` package carries a source generator that adds support for type-safe function definition generation. For more information, please check out [Create type-safe function](./Create-type-safe-function-call.md).

> [!NOTE]
> If you are using VSCode as your editor, you may need to restart the editor to see the generated code.

Firstly, import the required namespaces:
[!code-csharp[](../../sample/AutoGen.BasicSamples/CodeSnippet/OllamaCodeSnippet.cs?name=using_statement)]

Then, define a public partial class: `Function` with `GetWeather` method
[!code-csharp[](../../sample/AutoGen.BasicSamples/CodeSnippet/OllamaCodeSnippet.cs?name=weather_function)]

Then, create an @AutoGen.Ollama.OllamaChatAgent and register it with @AutoGen.Ollama.OllamaChatRequestMessageConnector so it can support @AutoGen.Core.ToolCallMessage and @AutoGen.Core.ToolCallResultMessage. These message types are necessary to use @AutoGen.Core.FunctionCallMiddleware, which provides support for processing and invoking function calls.

[!code-csharp[](../../sample/AutoGen.BasicSamples/CodeSnippet/OllamaCodeSnippet.cs?name=openai_chat_agent_get_weather_function_call)]

Then, create an @AutoGen.Core.FunctionCallMiddleware with `GetWeather` function and register it with the agent above. When creating the middleware, we also pass a `functionMap` to @AutoGen.Core.FunctionCallMiddleware, which means the function will be automatically invoked when the agent replies a `GetWeather` function call.

[!code-csharp[](../../sample/AutoGen.BasicSamples/CodeSnippet/OllamaCodeSnippet.cs?name=create_function_call_middleware)]

Finally, you can chat with the @AutoGen.Ollama.OllamaChatAgent and invoke the `GetWeather` function.

[!code-csharp[](../../sample/AutoGen.BasicSamples/CodeSnippet/OllamaCodeSnippet.cs?name=chat_agent_send_function_call)]
"""
logger.info("Firstly, import the required namespaces:")

logger.info("\n\n[DONE]", bright=True)