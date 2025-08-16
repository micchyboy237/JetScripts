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