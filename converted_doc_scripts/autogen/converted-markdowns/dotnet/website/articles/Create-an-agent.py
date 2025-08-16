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
## AssistantAgent

[`AssistantAgent`](../api/AutoGen.AssistantAgent.yml) is a built-in agent in `AutoGen` that acts as an AI assistant. It uses LLM to generate response to user input. It also supports function call if the underlying LLM model supports it (e.g. `gpt-3.5-turbo-0613`).

## Create an `AssistantAgent` using Ollama model.

[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/CodeSnippet/CreateAnAgent.cs?name=code_snippet_1)]

## Create an `AssistantAgent` using Azure Ollama model.

[!code-csharp[](../../samples/AgentChat/Autogen.Basic.Sample/CodeSnippet/CreateAnAgent.cs?name=code_snippet_2)]
"""
logger.info("## AssistantAgent")

logger.info("\n\n[DONE]", bright=True)