

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
The following example shows how to connect to third-party Ollama API using @AutoGen.Ollama.OllamaChatAgent.

[![](https://img.shields.io/badge/Open%20on%20Github-grey?logo=github)](https://github.com/microsoft/autogen/blob/main/dotnet/samples/AutoGen.Ollama.Sample/Connect_To_Ollama.cs)

## Overview
A lot of LLM applications/platforms support spinning up a chat server that is compatible with Ollama API, such as LM Studio, Ollama, Mistral etc. This means that you can connect to these servers using the @AutoGen.Ollama.OllamaChatAgent.

> [!NOTE]
> Some platforms might not support all the features of Ollama API. For example, Ollama does not support `function call` when using it's openai API according to its [document](https://github.com/ollama/ollama/blob/main/docs/openai.md#v1chatcompletions) (as of 2024/05/07).
> That means some of the features of Ollama API might not work as expected when using these platforms with the @AutoGen.Ollama.OllamaChatAgent.
> Please refer to the platform's documentation for more information.

## Prerequisites
- Install the following packages:
"""
logger.info("## Overview")

dotnet add package AutoGen.Ollama --version AUTOGEN_VERSION

"""
- Spin up a chat server that is compatible with Ollama API.
The following example uses Ollama as the chat server, and llama3 as the llm model.
"""
logger.info("The following example uses Ollama as the chat server, and llama3 as the llm model.")

ollama serve

"""
## Steps
- Import the required namespaces:
[!code-csharp[](../../samples/AutoGen.Ollama.Sample/Connect_To_Ollama.cs?name=using_statement)]

- Create a `CustomHttpClientHandler` class.

The `CustomHttpClientHandler` class is used to customize the HttpClientHandler. In this example, we override the `SendAsync` method to redirect the request to local Ollama server, which is running on `http://localhost:11434`.

[!code-csharp[](../../samples/AutoGen.Ollama.Sample/Connect_To_Ollama.cs?name=CustomHttpClientHandler)]

- Create an `OllamaChatAgent` instance and connect to the third-party API.

Then create an @AutoGen.Ollama.OllamaChatAgent instance and connect to the Ollama API from Ollama. You can customize the transport behavior of `OllamaClient` by passing a customized `HttpClientTransport` instance. In the customized `HttpClientTransport` instance, we pass the `CustomHttpClientHandler` we just created which redirects all openai chat requests to the local Ollama server.

[!code-csharp[](../../samples/AutoGen.Ollama.Sample/Connect_To_Ollama.cs?name=create_agent)]

- Chat with the `OllamaChatAgent`.
Finally, you can start chatting with the agent. In this example, we send a coding question to the agent and get the response.

[!code-csharp[](../../samples/AutoGen.Ollama.Sample/Connect_To_Ollama.cs?name=send_message)]

## Sample Output
The following is the sample output of the code snippet above:

![output](../images/articles/ConnectTo3PartyOllama/output.gif)
"""
logger.info("## Steps")

logger.info("\n\n[DONE]", bright=True)