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
The following example shows how to create a `MistralAITokenCounterMiddleware` @AutoGen.Core.IMiddleware and count the token usage when chatting with @AutoGen.Mistral.MistralClientAgent.

### Overview
To collect the token usage for the entire chat session, one easy solution is simply collect all the responses from agent and sum up the token usage for each response. To collect all the agent responses, we can create a middleware which simply saves all responses to a list and register it with the agent. To get the token usage information for each response, because in the example we are using @AutoGen.Mistral.MistralClientAgent, we can simply get the token usage from the response object.

> [!NOTE]
> You can find the complete example in the [Example13_MLXAgent_JsonMode](https://github.com/microsoft/autogen/tree/main/dotnet/samples/AgentChat/Autogen.Basic.Sample/Example14_MistralClientAgent_TokenCount.cs).

- Step 1: Adding using statement
[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/Example14_MistralClientAgent_TokenCount.cs?name=using_statements)]

- Step 2: Create a `MistralAITokenCounterMiddleware` class which implements @AutoGen.Core.IMiddleware. This middleware will collect all the responses from the agent and sum up the token usage for each response.
[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/Example14_MistralClientAgent_TokenCount.cs?name=token_counter_middleware)]

- Step 3: Create a `MistralClientAgent`
[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/Example14_MistralClientAgent_TokenCount.cs?name=create_mistral_client_agent)]

- Step 4: Register the `MistralAITokenCounterMiddleware` with the `MistralClientAgent`. Note that the order of each middlewares matters. The token counter middleware needs to be registered before `mistralMessageConnector` because it collects response only when the responding message type is `IMessage<ChatCompletionResponse>` while the `mistralMessageConnector` will convert `IMessage<ChatCompletionResponse>` to one of @AutoGen.Core.TextMessage, @AutoGen.Core.ToolCallMessage or @AutoGen.Core.ToolCallResultMessage.
[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/Example14_MistralClientAgent_TokenCount.cs?name=register_middleware)]

- Step 5: Chat with the `MistralClientAgent` and get the token usage information from the response object.
[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/Example14_MistralClientAgent_TokenCount.cs?name=chat_with_agent)]

### Output
When running the example, the completion token count will be printed to the console.
"""
logger.info("### Overview")

Completion token count: 1408 # might be different based on the response

logger.info("\n\n[DONE]", bright=True)