from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# gitty (Warning: WIP)

This is an AutoGen powered CLI that generates draft replies for issues and pull requests
to reduce maintenance overhead for open source projects.

Simple installation and CLI:

   ```bash
   gitty --repo microsoft/autogen issue 5212
   ```

*Important*: Install the dependencies and set Ollama API key:

   ```bash
   uv sync --all-extras
   source .venv/bin/activate
#    export OPENAI_API_KEY=sk-....
   ```
"""
logger.info("# gitty (Warning: WIP)")

logger.info("\n\n[DONE]", bright=True)