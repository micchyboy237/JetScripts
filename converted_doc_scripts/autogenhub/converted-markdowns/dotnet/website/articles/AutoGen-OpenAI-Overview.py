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
## AutoGen.MLX Overview

AutoGen.MLX provides the following agents over openai models:
- @AutoGen.MLX.MLXChatAgent: A slim wrapper agent over `MLXClient`. This agent only support `IMessage<ChatRequestMessage>` message type. To support more message types like @AutoGen.Core.TextMessage, register the agent with @AutoGen.MLX.MLXChatRequestMessageConnector.
- @AutoGen.MLX.GPTAgent: An agent that build on top of @AutoGen.MLX.MLXChatAgent with more message types support like @AutoGen.Core.TextMessage, @AutoGen.Core.ImageMessage, @AutoGen.Core.MultiModalMessage and function call support. Essentially, it is equivalent to @AutoGen.MLX.MLXChatAgent with @AutoGen.Core.FunctionCallMiddleware and @AutoGen.MLX.MLXChatRequestMessageConnector registered.

### Get start with AutoGen.MLX

To get start with AutoGen.MLX, firstly, follow the [installation guide](Installation.md) to make sure you add the AutoGen feed correctly. Then add `AutoGen.MLX` package to your project file.
"""
logger.info("## AutoGen.MLX Overview")

<ItemGroup>
    <PackageReference Include="AutoGen.MLX" Version="AUTOGEN_VERSION" />
</ItemGroup>

logger.info("\n\n[DONE]", bright=True)