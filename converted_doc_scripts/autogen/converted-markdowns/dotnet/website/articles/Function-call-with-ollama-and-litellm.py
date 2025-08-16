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
This example shows how to use function call with local LLM models where [Ollama](https://ollama.com/) as local model provider and [LiteLLM](https://docs.litellm.ai/docs/) proxy server which provides an openai-api compatible interface.

[![](https://img.shields.io/badge/Open%20on%20Github-grey?logo=github)](https://github.com/microsoft/autogen/blob/main/dotnet/samples/AutoGen.Ollama.Sample/Tool_Call_With_Ollama_And_LiteLLM.cs)

To run this example, the following prerequisites are required:
- Install [Ollama](https://ollama.com/) and [LiteLLM](https://docs.litellm.ai/docs/) on your local machine.
- A local model that supports function call. In this example `dolphincoder:latest` is used.

## Install Ollama and pull `dolphincoder:latest` model
First, install Ollama by following the instructions on the [Ollama website](https://ollama.com/).

After installing Ollama, pull the `dolphincoder:latest` model by running the following command:
"""
logger.info("## Install Ollama and pull `dolphincoder:latest` model")

ollama pull dolphincoder:latest

"""
## Install LiteLLM and start the proxy server

You can install LiteLLM by following the instructions on the [LiteLLM website](https://docs.litellm.ai/docs/).
"""
logger.info("## Install LiteLLM and start the proxy server")

pip install 'litellm[proxy]'

"""
Then, start the proxy server by running the following command:
"""
logger.info("Then, start the proxy server by running the following command:")

litellm --model ollama_chat/dolphincoder --port 4000

"""
This will start an openai-api compatible proxy server at `http://localhost:4000`. You can verify if the server is running by observing the following output in the terminal:
"""
logger.info("This will start an openai-api compatible proxy server at `http://localhost:4000`. You can verify if the server is running by observing the following output in the terminal:")

INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:4000 (Press CTRL+C to quit)

"""
## Install AutoGen and AutoGen.SourceGenerator
In your project, install the AutoGen and AutoGen.SourceGenerator package using the following command:
"""
logger.info("## Install AutoGen and AutoGen.SourceGenerator")

dotnet add package AutoGen
dotnet add package AutoGen.SourceGenerator

"""
The `AutoGen.SourceGenerator` package is used to automatically generate type-safe `FunctionContract` instead of manually defining them. For more information, please check out [Create type-safe function](Create-type-safe-function-call.md).

And in your project file, enable structural xml document support by setting the `GenerateDocumentationFile` property to `true`:
"""
logger.info("The `AutoGen.SourceGenerator` package is used to automatically generate type-safe `FunctionContract` instead of manually defining them. For more information, please check out [Create type-safe function](Create-type-safe-function-call.md).")

<PropertyGroup>
    <!-- This enables structural xml document support -->
    <GenerateDocumentationFile>true</GenerateDocumentationFile>
</PropertyGroup>

"""
## Define `WeatherReport` function and create @AutoGen.Core.FunctionCallMiddleware

Create a `public partial` class to host the methods you want to use in AutoGen agents. The method has to be a `public` instance method and its return type must be `Task<string>`. After the methods are defined, mark them with `AutoGen.Core.FunctionAttribute` attribute.

[!code-csharp[Define WeatherReport function](../../samples/AutoGen.Ollama.Sample/Tool_Call_With_Ollama_And_LiteLLM.cs?name=Function)]

Then create a @AutoGen.Core.FunctionCallMiddleware and add the `WeatherReport` function to the middleware. The middleware will pass the `FunctionContract` to the agent when generating a response, and process the tool call response when receiving a `ToolCallMessage`.
[!code-csharp[Define WeatherReport function](../../samples/AutoGen.Ollama.Sample/Tool_Call_With_Ollama_And_LiteLLM.cs?name=Create_tools)]

## Create @AutoGen.Ollama.OllamaChatAgent with `GetWeatherReport` tool and chat with it

Because LiteLLM proxy server is openai-api compatible, we can use @AutoGen.Ollama.OllamaChatAgent to connect to it as a third-party openai-api provider. The agent is also registered with a @AutoGen.Core.FunctionCallMiddleware which contains the `WeatherReport` tool. Therefore, the agent can call the `WeatherReport` tool when generating a response.

[!code-csharp[Create an agent with tools](../../samples/AutoGen.Ollama.Sample/Tool_Call_With_Ollama_And_LiteLLM.cs?name=Create_Agent)]

The reply from the agent will similar to the following:
"""
logger.info("## Define `WeatherReport` function and create @AutoGen.Core.FunctionCallMiddleware")

AggregateMessage from assistant
--------------------
ToolCallMessage:
ToolCallMessage from assistant
--------------------
- GetWeatherAsync: {"city": "new york"}
--------------------

ToolCallResultMessage:
ToolCallResultMessage from assistant
--------------------
- GetWeatherAsync: The weather in new york is 72 degrees and sunny.
--------------------

logger.info("\n\n[DONE]", bright=True)