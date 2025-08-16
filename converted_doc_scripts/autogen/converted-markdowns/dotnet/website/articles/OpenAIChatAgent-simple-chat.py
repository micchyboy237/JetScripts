from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
The following example shows how to create an @AutoGen.Ollama.OllamaChatAgent and chat with it.

Firsly, import the required namespaces:
[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/CodeSnippet/OllamaCodeSnippet.cs?name=using_statement)]

Then, create an @AutoGen.Ollama.OllamaChatAgent and chat with it:
[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/CodeSnippet/OllamaCodeSnippet.cs?name=create_openai_chat_agent)]

@AutoGen.Ollama.OllamaChatAgent also supports streaming chat via @AutoGen.Core.IAgent.GenerateStreamingReplyAsync*.

[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/CodeSnippet/OllamaCodeSnippet.cs?name=create_openai_chat_agent_streaming)]
"""
logger.info("The following example shows how to create an @AutoGen.Ollama.OllamaChatAgent and chat with it.")

logger.info("\n\n[DONE]", bright=True)