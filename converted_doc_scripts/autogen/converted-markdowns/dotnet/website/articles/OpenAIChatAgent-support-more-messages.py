from jet.logger import CustomLogger
import os
import shutil
import the required namespaces:


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
By default, @AutoGen.Ollama.OllamaChatAgent only supports the @AutoGen.Core.IMessage<T> type where `T` is original request or response message from `Azure.AI.Ollama`. To support more AutoGen built-in message types like @AutoGen.Core.TextMessage, @AutoGen.Core.ImageMessage, @AutoGen.Core.MultiModalMessage and so on, you can register the agent with @AutoGen.Ollama.OllamaChatRequestMessageConnector. The @AutoGen.Ollama.OllamaChatRequestMessageConnector will convert the message from AutoGen built-in message types to `Azure.AI.Ollama.ChatRequestMessage` and vice versa.

[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/CodeSnippet/OllamaCodeSnippet.cs?name=using_statement)]

[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/CodeSnippet/OllamaCodeSnippet.cs?name=register_openai_chat_message_connector)]
"""
logger.info("By default, @AutoGen.Ollama.OllamaChatAgent only supports the @AutoGen.Core.IMessage<T> type where `T` is original request or response message from `Azure.AI.Ollama`. To support more AutoGen built-in message types like @AutoGen.Core.TextMessage, @AutoGen.Core.ImageMessage, @AutoGen.Core.MultiModalMessage and so on, you can register the agent with @AutoGen.Ollama.OllamaChatRequestMessageConnector. The @AutoGen.Ollama.OllamaChatRequestMessageConnector will convert the message from AutoGen built-in message types to `Azure.AI.Ollama.ChatRequestMessage` and vice versa.")

logger.info("\n\n[DONE]", bright=True)