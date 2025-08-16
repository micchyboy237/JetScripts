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
This example shows how to use @AutoGen.Ollama.OllamaAgent to connect to Ollama server and chat with LLaVA model.

To run this example, you need to have an Ollama server running aside and have `llama3:latest` model installed. For how to setup an Ollama server, please refer to [Ollama](https://ollama.com/).

> [!NOTE]
> You can find the complete sample code [here](https://github.com/microsoft/autogen/blob/main/dotnet/samples/AutoGen.Ollama.Sample/Chat_With_LLaMA.cs)

### Step 1: Install AutoGen.Ollama

First, install the AutoGen.Ollama package using the following command:
"""
logger.info("### Step 1: Install AutoGen.Ollama")

dotnet add package AutoGen.Ollama

"""
For how to install from nightly build, please refer to [Installation](../Installation.md).

### Step 2: Add using statement

[!code-csharp[](../../../samples/AutoGen.Ollama.Sample/Chat_With_LLaMA.cs?name=Using)]

### Step 3: Create and chat @AutoGen.Ollama.OllamaAgent

In this step, we create an @AutoGen.Ollama.OllamaAgent and connect it to the Ollama server.

[!code-csharp[](../../../samples/AutoGen.Ollama.Sample/Chat_With_LLaMA.cs?name=Create_Ollama_Agent)]
"""
logger.info("### Step 2: Add using statement")

logger.info("\n\n[DONE]", bright=True)