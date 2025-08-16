from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Async Human-in-the-Loop Example

An example showing human-in-the-loop which waits for human input before making the tool call.

## Prerequisites

First, you need a shell with AutoGen core and required dependencies installed.
"""
logger.info("# Async Human-in-the-Loop Example")

pip install "autogen-ext[openai,azure]" "pyyaml"

"""
## Model Configuration

The model configuration should defined in a `model_config.yml` file.
Use `model_config_template.yml` as a template.

## Running the example
"""
logger.info("## Model Configuration")

python main.py

logger.info("\n\n[DONE]", bright=True)