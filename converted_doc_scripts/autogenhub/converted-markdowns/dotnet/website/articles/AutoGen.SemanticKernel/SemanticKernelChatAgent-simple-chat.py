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
`AutoGen.SemanticKernel` provides built-in support for `ChatCompletionAgent` via @AutoGen.SemanticKernel.SemanticKernelChatCompletionAgent. By default the @AutoGen.SemanticKernel.SemanticKernelChatCompletionAgent only supports the original `ChatMessageContent` type via `IMessage<ChatMessageContent>`. To support more AutoGen built-in message types like @AutoGen.Core.TextMessage, @AutoGen.Core.ImageMessage, @AutoGen.Core.MultiModalMessage, you can register the agent with @AutoGen.SemanticKernel.SemanticKernelChatMessageContentConnector. The @AutoGen.SemanticKernel.SemanticKernelChatMessageContentConnector will convert the message from AutoGen built-in message types to `ChatMessageContent` and vice versa.

The following step-by-step example shows how to create an @AutoGen.SemanticKernel.SemanticKernelChatCompletionAgent and chat with it:

> [!NOTE]
> You can find the complete sample code [here](https://github.com/autogenhub/autogen/blob/main/dotnet/sample/AutoGen.SemanticKernel.Sample/Create_Semantic_Kernel_Chat_Agent.cs).

### Step 1: add using statement
[!code-csharp[](../../../sample/AutoGen.SemanticKernel.Sample/Create_Semantic_Kernel_Chat_Agent.cs?name=Using)]

### Step 2: create kernel
[!code-csharp[](../../../sample/AutoGen.SemanticKernel.Sample/Create_Semantic_Kernel_Chat_Agent.cs?name=Create_Kernel)]

### Step 3: create ChatCompletionAgent
[!code-csharp[](../../../sample/AutoGen.SemanticKernel.Sample/Create_Semantic_Kernel_Chat_Agent.cs?name=Create_ChatCompletionAgent)]

### Step 4: create @AutoGen.SemanticKernel.SemanticKernelChatCompletionAgent
In this step, we create an @AutoGen.SemanticKernel.SemanticKernelChatCompletionAgent and register it with @AutoGen.SemanticKernel.SemanticKernelChatMessageContentConnector. The @AutoGen.SemanticKernel.SemanticKernelChatMessageContentConnector will convert the message from AutoGen built-in message types to `ChatMessageContent` and vice versa.
[!code-csharp[](../../../sample/AutoGen.SemanticKernel.Sample/Create_Semantic_Kernel_Chat_Agent.cs?name=Create_SemanticKernelChatCompletionAgent)]

### Step 5: chat with @AutoGen.SemanticKernel.SemanticKernelChatCompletionAgent
[!code-csharp[](../../../sample/AutoGen.SemanticKernel.Sample/Create_Semantic_Kernel_Chat_Agent.cs?name=Send_Message)]
"""
logger.info("### Step 1: add using statement")

logger.info("\n\n[DONE]", bright=True)