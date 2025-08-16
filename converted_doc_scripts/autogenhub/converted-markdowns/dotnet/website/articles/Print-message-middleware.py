from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
@AutoGen.Core.PrintMessageMiddleware is a built-in @AutoGen.Core.IMiddleware that pretty print @AutoGen.Core.IMessage to console.

> [!NOTE]
> @AutoGen.Core.PrintMessageMiddleware support the following @AutoGen.Core.IMessage types:
> - @AutoGen.Core.TextMessage
> - @AutoGen.Core.MultiModalMessage
> - @AutoGen.Core.ToolCallMessage
> - @AutoGen.Core.ToolCallResultMessage
> - @AutoGen.Core.Message
> - (streaming) @AutoGen.Core.TextMessageUpdate
> - (streaming) @AutoGen.Core.ToolCallMessageUpdate

## Use @AutoGen.Core.PrintMessageMiddleware in an agent
You can use @AutoGen.Core.PrintMessageMiddlewareExtension.RegisterPrintMessage* to register the @AutoGen.Core.PrintMessageMiddleware to an agent.

[!code-csharp[](../../sample/AutoGen.BasicSamples/CodeSnippet/PrintMessageMiddlewareCodeSnippet.cs?name=PrintMessageMiddleware)]

@AutoGen.Core.PrintMessageMiddlewareExtension.RegisterPrintMessage* will format the message and print it to console
![image](../images/articles/PrintMessageMiddleware/printMessage.png)

## Streaming message support

@AutoGen.Core.PrintMessageMiddleware also supports streaming message types like @AutoGen.Core.TextMessageUpdate and @AutoGen.Core.ToolCallMessageUpdate. If you register @AutoGen.Core.PrintMessageMiddleware to a @AutoGen.Core.IStreamingAgent, it will format the streaming message and print it to console if the message is of supported type.

[!code-csharp[](../../sample/AutoGen.BasicSamples/CodeSnippet/PrintMessageMiddlewareCodeSnippet.cs?name=print_message_streaming)]

![image](../images/articles/PrintMessageMiddleware/streamingoutput.gif)
"""
logger.info("## Use @AutoGen.Core.PrintMessageMiddleware in an agent")

logger.info("\n\n[DONE]", bright=True)