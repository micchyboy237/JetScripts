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
# AutoGen.Gemini Overview

AutoGen.Gemini is a package that provides seamless integration with Google Gemini. It provides the following agent:

- @AutoGen.Gemini.GeminiChatAgent: The agent that connects to Google Gemini or Vertex AI Gemini. It supports chat, multi-modal chat, and function call.

AutoGen.Gemini also provides the following middleware:
- @AutoGen.Gemini.GeminiMessageConnector: The middleware that converts the Gemini message to AutoGen built-in message type.

## Examples

You can find more examples under the [gemini sample project](https://github.com/microsoft/autogen/tree/main/dotnet/samples/AutoGen.Gemini.Sample)
"""
logger.info("# AutoGen.Gemini Overview")

logger.info("\n\n[DONE]", bright=True)