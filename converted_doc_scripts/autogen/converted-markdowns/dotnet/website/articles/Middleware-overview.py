from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
`Middleware` is a key feature in AutoGen.Net that enables you to customize the behavior of @AutoGen.Core.IAgent.GenerateReplyAsync*. It's similar to the middleware concept in ASP.Net and is widely used in AutoGen.Net for various scenarios, such as function call support, converting message of different types, print message, gather user input, etc.

Here are a few examples of how middleware is used in AutoGen.Net:
- @AutoGen.AssistantAgent is essentially an agent with @AutoGen.Core.FunctionCallMiddleware, @AutoGen.HumanInputMiddleware and default reply middleware.
- @AutoGen.Ollama.GPTAgent is essentially an @AutoGen.Ollama.OllamaChatAgent with @AutoGen.Core.FunctionCallMiddleware and @AutoGen.Ollama.OllamaChatRequestMessageConnector.

## Use middleware in an agent
To use middleware in an existing agent, you can either create a @AutoGen.Core.MiddlewareAgent on top of the original agent or register middleware functions to the original agent.

### Create @AutoGen.Core.MiddlewareAgent on top of the original agent
[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/CodeSnippet/MiddlewareAgentCodeSnippet.cs?name=create_middleware_agent_with_original_agent)]

### Register middleware functions to the original agent
[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/CodeSnippet/MiddlewareAgentCodeSnippet.cs?name=register_middleware_agent)]

## Short-circuit the next agent
The example below shows how to short-circuit the inner agent

[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/CodeSnippet/MiddlewareAgentCodeSnippet.cs?name=short_circuit_middleware_agent)]

> [!Note]
> When multiple middleware functions are registered, the order of middleware functions is first registered, last invoked.

## Streaming middleware
You can also modify the behavior of @AutoGen.Core.IStreamingAgent.GenerateStreamingReplyAsync* by registering streaming middleware to it. One example is @AutoGen.Ollama.OllamaChatRequestMessageConnector which converts `StreamingChatCompletionsUpdate` to one of `AutoGen.Core.TextMessageUpdate` or `AutoGen.Core.ToolCallMessageUpdate`.

[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/CodeSnippet/MiddlewareAgentCodeSnippet.cs?name=register_streaming_middleware)]
"""
logger.info("## Use middleware in an agent")

logger.info("\n\n[DONE]", bright=True)