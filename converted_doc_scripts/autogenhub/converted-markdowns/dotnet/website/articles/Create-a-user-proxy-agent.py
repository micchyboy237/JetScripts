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
## UserProxyAgent

[`UserProxyAgent`](../api/AutoGen.UserProxyAgent.yml) is a special type of agent that can be used to proxy user input to another agent or group of agents. It supports the following human input modes:
- `ALWAYS`: Always ask user for input.
- `NEVER`: Never ask user for input. In this mode, the agent will use the default response (if any) to respond to the message. Or using underlying LLM model to generate response if provided.
- `AUTO`: Only ask user for input when conversation is terminated by the other agent(s). Otherwise, use the default response (if any) to respond to the message. Or using underlying LLM model to generate response if provided.

> [!TIP]
> You can also set up `humanInputMode` when creating `AssistantAgent` to enable/disable human input. `UserProxyAgent` is equivalent to `AssistantAgent` with `humanInputMode` set to `ALWAYS`. Similarly, `AssistantAgent` is equivalent to `UserProxyAgent` with `humanInputMode` set to `NEVER`.

### Create a `UserProxyAgent` with `HumanInputMode` set to `ALWAYS`

[!code-csharp[](../../sample/AutoGen.BasicSamples/CodeSnippet/UserProxyAgentCodeSnippet.cs?name=code_snippet_1)]

When running the code, the user proxy agent will ask user for input and use the input as response.
![code output](../images/articles/CreateUserProxyAgent/image-1.png)
"""
logger.info("## UserProxyAgent")

logger.info("\n\n[DONE]", bright=True)