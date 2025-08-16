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
The following example shows how to enable JSON mode in @AutoGen.MLX.MLXChatAgent.

[![](https://img.shields.io/badge/Open%20on%20Github-grey?logo=github)](https://github.com/microsoft/autogen/blob/main/dotnet/samples/AutoGen.MLX.Sample/Use_Json_Mode.cs)

## What is JSON mode?
JSON mode is a new feature in MLX which allows you to instruct model to always respond with a valid JSON object. This is useful when you want to constrain the model output to JSON format only.

> [!NOTE]
> Currently, JOSN mode is only supported by `gpt-4-turbo-preview` and `gpt-3.5-turbo-0125`. For more information (and limitations) about JSON mode, please visit [MLX API documentation](https://platform.openai.com/docs/guides/text-generation/json-mode).

## How to enable JSON mode in MLXChatAgent.

To enable JSON mode for @AutoGen.MLX.MLXChatAgent, set `responseFormat` to `ChatCompletionsResponseFormat.JsonObject` when creating the agent. Note that when enabling JSON mode, you also need to instruct the agent to output JSON format in its system message.

[!code-csharp[](../../samples/AutoGen.MLX.Sample/Use_Json_Mode.cs?name=create_agent)]

After enabling JSON mode, the `openAIClientAgent` will always respond in JSON format when it receives a message.

[!code-csharp[](../../samples/AutoGen.MLX.Sample/Use_Json_Mode.cs?name=chat_with_agent)]

When running the example, the output from `openAIClientAgent` will be a valid JSON object which can be parsed as `Person` class defined below. Note that in the output, the `address` field is missing because the address information is not provided in user input.

[!code-csharp[](../../samples/AutoGen.MLX.Sample/Use_Json_Mode.cs?name=person_class)]

The output will be:
"""
logger.info("## What is JSON mode?")

Name: John
Age: 25
Done

logger.info("\n\n[DONE]", bright=True)