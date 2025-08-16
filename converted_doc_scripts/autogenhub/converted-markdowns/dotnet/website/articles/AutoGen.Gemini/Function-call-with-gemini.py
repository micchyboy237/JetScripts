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
This example shows how to use @AutoGen.Gemini.GeminiChatAgent to make function call. This example is modified from [gemini-api function call example](https://ai.google.dev/gemini-api/docs/function-calling)

To run this example, you need to have a project on Google Cloud with access to Vertex AI API. For more information please refer to [Google Vertex AI](https://cloud.google.com/vertex-ai/docs).


> [!NOTE]
> You can find the complete sample code [here](https://github.com/autogenhub/autogen/blob/main/dotnet/sample/AutoGen.Gemini.Sample/Function_Call_With_Gemini.cs)

### Step 1: Install AutoGen.Gemini and AutoGen.SourceGenerator

First, install the AutoGen.Gemini package using the following command:
"""
logger.info("### Step 1: Install AutoGen.Gemini and AutoGen.SourceGenerator")

dotnet add package AutoGen.Gemini
dotnet add package AutoGen.SourceGenerator

"""
The AutoGen.SourceGenerator package is required to generate the @AutoGen.Core.FunctionContract. For more information, please refer to [Create-type-safe-function-call](../Create-type-safe-function-call.md)

### Step 2: Add using statement
[!code-csharp[](../../../sample/AutoGen.Gemini.Sample/Function_call_with_gemini.cs?name=Using)]

### Step 3: Create `MovieFunction`

[!code-csharp[](../../../sample/AutoGen.Gemini.Sample/Function_call_with_gemini.cs?name=MovieFunction)]

### Step 4: Create a Gemini agent

[!code-csharp[](../../../sample/AutoGen.Gemini.Sample/Function_call_with_gemini.cs?name=Create_Gemini_Agent)]

### Step 5: Single turn function call

[!code-csharp[](../../../sample/AutoGen.Gemini.Sample/Function_call_with_gemini.cs?name=Single_turn)]

### Step 6: Multi-turn function call

[!code-csharp[](../../../sample/AutoGen.Gemini.Sample/Function_call_with_gemini.cs?name=Multi_turn)]
"""
logger.info("### Step 2: Add using statement")

logger.info("\n\n[DONE]", bright=True)