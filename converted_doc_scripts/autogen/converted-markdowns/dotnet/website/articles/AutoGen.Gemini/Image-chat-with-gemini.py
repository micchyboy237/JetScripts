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
This example shows how to use @AutoGen.Gemini.GeminiChatAgent for image chat with Gemini model.

To run this example, you need to have a project on Google Cloud with access to Vertex AI API. For more information please refer to [Google Vertex AI](https://cloud.google.com/vertex-ai/docs).


> [!NOTE]
> You can find the complete sample code [here](https://github.com/microsoft/autogen/blob/main/dotnet/samples/AutoGen.Gemini.Sample/Image_Chat_With_Vertex_Gemini.cs)

### Step 1: Install AutoGen.Gemini

First, install the AutoGen.Gemini package using the following command:
"""
logger.info("### Step 1: Install AutoGen.Gemini")

dotnet add package AutoGen.Gemini

"""
### Step 2: Add using statement
[!code-csharp[](../../../samples/AutoGen.Gemini.Sample/Image_Chat_With_Vertex_Gemini.cs?name=Using)]

### Step 3: Create a Gemini agent

[!code-csharp[](../../../samples/AutoGen.Gemini.Sample/Image_Chat_With_Vertex_Gemini.cs?name=Create_Gemini_Agent)]

### Step 4: Send image to Gemini
[!code-csharp[](../../../samples/AutoGen.Gemini.Sample/Image_Chat_With_Vertex_Gemini.cs?name=Send_Image_Request)]
"""
logger.info("### Step 2: Add using statement")

logger.info("\n\n[DONE]", bright=True)