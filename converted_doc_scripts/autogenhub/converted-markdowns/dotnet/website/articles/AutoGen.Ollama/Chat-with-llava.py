

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
This sample shows how to use @AutoGen.Ollama.OllamaAgent to chat with LLaVA model.

To run this example, you need to have an Ollama server running aside and have `llava:latest` model installed. For how to setup an Ollama server, please refer to [Ollama](https://ollama.com/).

> [!NOTE]
> You can find the complete sample code [here](https://github.com/autogenhub/autogen/blob/main/dotnet/sample/AutoGen.Ollama.Sample/Chat_With_LLaVA.cs)

### Step 1: Install AutoGen.Ollama

First, install the AutoGen.Ollama package using the following command:
"""
logger.info("### Step 1: Install AutoGen.Ollama")

dotnet add package AutoGen.Ollama

"""
For how to install from nightly build, please refer to [Installation](../Installation.md).

### Step 2: Add using statement

[!code-csharp[](../../../sample/AutoGen.Ollama.Sample/Chat_With_LLaVA.cs?name=Using)]

### Step 3: Create @AutoGen.Ollama.OllamaAgent

[!code-csharp[](../../../sample/AutoGen.Ollama.Sample/Chat_With_LLaVA.cs?name=Create_Ollama_Agent)]

### Step 4: Start MultiModal Chat
LLaVA is a multimodal model that supports both text and image inputs. In this step, we create an image message along with a question about the image.

[!code-csharp[](../../../sample/AutoGen.Ollama.Sample/Chat_With_LLaVA.cs?name=Send_Message)]
"""
logger.info("### Step 2: Add using statement")

logger.info("\n\n[DONE]", bright=True)