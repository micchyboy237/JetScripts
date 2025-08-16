

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
You can chat with @AutoGen.SemanticKernel.SemanticKernelAgent using both streaming and non-streaming methods and use native `ChatMessageContent` type via `IMessage<ChatMessageContent>`.

The following example shows how to create an @AutoGen.SemanticKernel.SemanticKernelAgent and chat with it using non-streaming method:

[!code-csharp[](../../../samples/AgentChat/Autogen.Basic.Sample/CodeSnippet/SemanticKernelCodeSnippet.cs?name=create_semantic_kernel_agent)]

@AutoGen.SemanticKernel.SemanticKernelAgent also supports streaming chat via @AutoGen.Core.IStreamingAgent.GenerateStreamingReplyAsync*.

[!code-csharp[](../../../samples/AgentChat/Autogen.Basic.Sample/CodeSnippet/SemanticKernelCodeSnippet.cs?name=create_semantic_kernel_agent_streaming)]
"""
logger.info("You can chat with @AutoGen.SemanticKernel.SemanticKernelAgent using both streaming and non-streaming methods and use native `ChatMessageContent` type via `IMessage<ChatMessageContent>`.")

logger.info("\n\n[DONE]", bright=True)