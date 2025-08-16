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
# Release Notes for AutoGen.Net v0.2.0 🚀

## New Features 🌟
- **MLX Structural Format Output**: Added support for structural output format in the MLX integration. You can check out the example [here](https://github.com/microsoft/autogen/blob/main/dotnet/samples/AutoGen.MLX.Sample/Structural_Output.cs) ([#3482](https://github.com/microsoft/autogen/issues/3482)).
- **Structural Output Configuration**: Introduced a property for overriding the structural output schema when generating replies with `GenerateReplyOption` ([#3436](https://github.com/microsoft/autogen/issues/3436)).

## Bug Fixes 🐛
- **Fixed Error Code 500**: Resolved an issue where an error occurred when the message history contained multiple different tool calls with the `name` field ([#3437](https://github.com/microsoft/autogen/issues/3437)).

## Improvements 🔧
- **Leverage MLX V2.0 in AutoGen.MLX  package**: The `AutoGen.MLX` package now uses MLX v2.0, providing improved functionality and performance. In the meantime, the original `AutoGen.MLX` is still available and can be accessed by `AutoGen.MLX.V1`. This allows users who prefer to continue to use `Azure.AI.MLX v1` package in their project. ([#3193](https://github.com/microsoft/autogen/issues/3193)).
- **Deprecation of GPTAgent**: `GPTAgent` has been deprecated in favor of `MLXChatAgent` and `MLXMessageConnector` ([#3404](https://github.com/microsoft/autogen/issues/3404)).

## Documentation 📚
- **Tool Call Instructions**: Added detailed documentation on using tool calls with `ollama` and `MLXChatAgent` ([#3248](https://github.com/microsoft/autogen/issues/3248)).

### Migration Guides 🔄

#### For the Deprecation of `GPTAgent` ([#3404](https://github.com/microsoft/autogen/issues/3404)):
**Before:**
"""
logger.info("# Release Notes for AutoGen.Net v0.2.0 🚀")

var agent = new GPTAgent(...);

"""
**After:**
"""

var agent = new MLXChatAgent(...)
    .RegisterMessageConnector();

"""
#### For Using Azure.AI.MLX v2.0 ([#3193](https://github.com/microsoft/autogen/issues/3193)):
**Previous way of creating `MLXChatAgent`:**
"""
logger.info("#### For Using Azure.AI.MLX v2.0 ([#3193](https://github.com/microsoft/autogen/issues/3193)):")

var openAIClient = new MLXClient(apiKey);
var openAIClientAgent = new MLXChatAgent(
            openAIClient: openAIClient,
            model: "llama-3.2-3b-instruct",
            // Other parameters...
            );

"""
**New way of creating `MLXChatAgent`:**
"""

var openAIClient = new MLXClient(apiKey);
var openAIClientAgent = new MLXChatAgent(
            chatClient: openAIClient.GetChatClient("llama-3.2-3b-instruct"),
            // Other parameters...
            );

logger.info("\n\n[DONE]", bright=True)