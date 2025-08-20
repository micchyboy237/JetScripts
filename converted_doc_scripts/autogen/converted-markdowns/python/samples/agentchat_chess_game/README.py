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
# AgentChat Chess Game

This is a simple chess game that you can play with an AI agent.

## Setup

Install the `chess` package with the following command:
"""
logger.info("# AgentChat Chess Game")

pip install "chess"

"""
To use MLX models or models hosted on MLX-compatible API endpoints,
you need to install the `autogen-ext[openai]` package. You can install it with the following command:
"""
logger.info("To use MLX models or models hosted on MLX-compatible API endpoints,")

pip install "autogen-ext[openai]"

"""
To run this sample, you will need to install the following packages:
"""
logger.info(
    "To run this sample, you will need to install the following packages:")

pip install - U autogen-agentchat pyyaml

"""
Create a new file named `model_config.yaml` in the the same directory as the script
to configure the model you want to use.

For example, to use `gpt-4o` model from MLX, you can use the following configuration:
"""
logger.info(
    "Create a new file named `model_config.yaml` in the the same directory as the script")

provider: jet.llm.mlx.autogen_ext.mlx_chat_completion_client.MLXAutogenChatLLMAdapter
config:
    model: gpt-4o
#   api_key: replace with your API key or skip it if you have environment variable OPENAI_API_KEY set

"""
To use `o3-mini-2025-01-31` model from MLX, you can use the following configuration:
"""
logger.info(
    "To use `o3-mini-2025-01-31` model from MLX, you can use the following configuration:")

provider: jet.llm.mlx.autogen_ext.mlx_chat_completion_client.MLXAutogenChatLLMAdapter
config:
    model: o3-mini-2025-01-31
#   api_key: replace with your API key or skip it if you have environment variable OPENAI_API_KEY set

"""
To use a locally hosted DeepSeek-R1:8b model using Ollama throught its compatibility endpoint,
you can use the following configuration:
"""
logger.info(
    "To use a locally hosted DeepSeek-R1:8b model using Ollama throught its compatibility endpoint,")

provider: jet.llm.mlx.autogen_ext.mlx_chat_completion_client.MLXAutogenChatLLMAdapter
config:
    model: deepseek-r1: 8b
    base_url: http: // localhost: 11434/v1
    api_key: ollama
    model_info:
        function_calling: false
        json_output: false
        vision: false
        family: r1

"""
For more information on how to configure the model and use other providers,
please refer to the [Models documentation](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/models.html).

## Run

Run the following command to start the game:
"""
logger.info("## Run")

python main.py

"""
By default, the game will use a random agent to play against the AI agent.
You can enable human vs AI mode by setting the `--human` flag:
"""
logger.info(
    "By default, the game will use a random agent to play against the AI agent.")

python main.py - -human

logger.info("\n\n[DONE]", bright=True)
