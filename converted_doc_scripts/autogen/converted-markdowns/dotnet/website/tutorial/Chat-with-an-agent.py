from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
This tutorial shows how to generate response using an @AutoGen.Core.IAgent by taking @AutoGen.Ollama.OllamaChatAgent as an example.

> [!NOTE]
> AutoGen.Net provides the following agents to connect to different LLM platforms. Generating responses using these agents is similar to the example shown below.
> - @AutoGen.Ollama.OllamaChatAgent
> - @AutoGen.SemanticKernel.SemanticKernelAgent
> - @AutoGen.LMStudio.LMStudioAgent
> - @AutoGen.Mistral.MistralClientAgent
> - @AutoGen.Anthropic.AnthropicClientAgent
> - @AutoGen.Ollama.OllamaAgent
> - @AutoGen.Gemini.GeminiChatAgent

> [!NOTE]
> The complete code example can be found in [Chat_With_Agent.cs](https://github.com/microsoft/autogen/blob/main/dotnet/samples/AgentChat/Autogen.Basic.Sample/GettingStart/Chat_With_Agent.cs)

## Step 1: Install AutoGen

First, install the AutoGen package using the following command:
"""
logger.info("## Step 1: Install AutoGen")

dotnet add package AutoGen

"""
## Step 2: add Using Statements

[!code-csharp[Using Statements](../../samples/AgentChat/Autogen.Basic.Sample/GettingStart/Chat_With_Agent.cs?name=Using)]

## Step 3: Create an @AutoGen.Ollama.OllamaChatAgent

> [!NOTE]
> The @AutoGen.Ollama.Extension.OllamaAgentExtension.RegisterMessageConnector* method registers an @AutoGen.Ollama.OllamaChatRequestMessageConnector middleware which converts Ollama message types to AutoGen message types. This step is necessary when you want to use AutoGen built-in message types like @AutoGen.Core.TextMessage, @AutoGen.Core.ImageMessage, etc.
> For more information, see [Built-in-messages](../articles/Built-in-messages.md)

[!code-csharp[Create an OllamaChatAgent](../../samples/AgentChat/Autogen.Basic.Sample/GettingStart/Chat_With_Agent.cs?name=Create_Agent)]

## Step 4: Generate Response
To generate response, you can use one of the overloaded method of @AutoGen.Core.AgentExtension.SendAsync* method. The following code shows how to generate response with text message:

[!code-csharp[Generate Response](../../samples/AgentChat/Autogen.Basic.Sample/GettingStart/Chat_With_Agent.cs?name=Chat_With_Agent)]

To generate response with chat history, you can pass the chat history to the @AutoGen.Core.AgentExtension.SendAsync* method:

[!code-csharp[Generate Response with Chat History](../../samples/AgentChat/Autogen.Basic.Sample/GettingStart/Chat_With_Agent.cs?name=Chat_With_History)]

To streamingly generate response, use @AutoGen.Core.IStreamingAgent.GenerateStreamingReplyAsync*

[!code-csharp[Generate Streaming Response](../../samples/AgentChat/Autogen.Basic.Sample/GettingStart/Chat_With_Agent.cs?name=Streaming_Chat)]

## Further Reading
- [Chat with google gemini](../articles/AutoGen.Gemini/Chat-with-google-gemini.md)
- [Chat with vertex gemini](../articles/AutoGen.Gemini/Chat-with-vertex-gemini.md)
- [Chat with Ollama](../articles/AutoGen.Ollama/Chat-with-llama.md)
- [Chat with Semantic Kernel Agent](../articles/AutoGen.SemanticKernel/SemanticKernelAgent-simple-chat.md)
"""
logger.info("## Step 2: add Using Statements")

logger.info("\n\n[DONE]", bright=True)