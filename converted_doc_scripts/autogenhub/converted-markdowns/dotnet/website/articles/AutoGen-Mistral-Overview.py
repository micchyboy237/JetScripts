from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
## AutoGen.Mistral overview

AutoGen.Mistral provides the following agent(s) to connect to [Mistral.AI](https://mistral.ai/) platform.
- @AutoGen.Mistral.MistralClientAgent: A slim wrapper agent over @AutoGen.Mistral.MistralClient.

### Get started with AutoGen.Mistral

To get started with AutoGen.Mistral, follow the [installation guide](Installation.md) to make sure you add the AutoGen feed correctly. Then add the `AutoGen.Mistral` package to your project file.
"""
logger.info("## AutoGen.Mistral overview")

dotnet add package AutoGen.Mistral

"""
>[!NOTE]
> You need to provide an api-key to use Mistral models which will bring additional cost while using. you can get the api key from [Mistral.AI](https://mistral.ai/).

### Example

Import the required namespace
[!code-csharp[](../../sample/AutoGen.BasicSamples/CodeSnippet/MistralAICodeSnippet.cs?name=using_statement)]

Create a @AutoGen.Mistral.MistralClientAgent and start chatting!
[!code-csharp[](../../sample/AutoGen.BasicSamples/CodeSnippet/MistralAICodeSnippet.cs?name=create_mistral_agent)]

Use @AutoGen.Core.IStreamingAgent.GenerateStreamingReplyAsync* to stream the chat completion.
[!code-csharp[](../../sample/AutoGen.BasicSamples/CodeSnippet/MistralAICodeSnippet.cs?name=streaming_chat)]
"""
logger.info("### Example")

logger.info("\n\n[DONE]", bright=True)