from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
## AssistantAgent

[`AssistantAgent`](../api/AutoGen.AssistantAgent.yml) is a built-in agent in `AutoGen` that acts as an AI assistant. It uses LLM to generate response to user input. It also supports function call if the underlying LLM model supports it (e.g. `gpt-3.5-turbo-0613`).

## Create an `AssistantAgent` using Ollama model.

[!code-csharp[](../../sample/AutoGen.BasicSamples/CodeSnippet/CreateAnAgent.cs?name=code_snippet_1)]

## Create an `AssistantAgent` using Azure Ollama model.

[!code-csharp[](../../sample/AutoGen.BasicSamples/CodeSnippet/CreateAnAgent.cs?name=code_snippet_2)]
"""
logger.info("## AssistantAgent")

logger.info("\n\n[DONE]", bright=True)