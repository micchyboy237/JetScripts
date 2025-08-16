

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
## AutoGen.Ollama Overview

AutoGen.Ollama provides the following agents over openai models:
- @AutoGen.Ollama.OllamaChatAgent: A slim wrapper agent over `OllamaClient`. This agent only support `IMessage<ChatRequestMessage>` message type. To support more message types like @AutoGen.Core.TextMessage, register the agent with @AutoGen.Ollama.OllamaChatRequestMessageConnector.
- @AutoGen.Ollama.GPTAgent: An agent that build on top of @AutoGen.Ollama.OllamaChatAgent with more message types support like @AutoGen.Core.TextMessage, @AutoGen.Core.ImageMessage, @AutoGen.Core.MultiModalMessage and function call support. Essentially, it is equivalent to @AutoGen.Ollama.OllamaChatAgent with @AutoGen.Core.FunctionCallMiddleware and @AutoGen.Ollama.OllamaChatRequestMessageConnector registered.

### Get start with AutoGen.Ollama

To get start with AutoGen.Ollama, firstly, follow the [installation guide](Installation.md) to make sure you add the AutoGen feed correctly. Then add `AutoGen.Ollama` package to your project file.
"""
logger.info("## AutoGen.Ollama Overview")

<ItemGroup>
    <PackageReference Include="AutoGen.Ollama" Version="AUTOGEN_VERSION" />
</ItemGroup>

logger.info("\n\n[DONE]", bright=True)