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
# Ollama

![Ollama Example](img/ecosystem-ollama.png)

[Ollama](https://ollama.com/) allows the users to run open-source large language models, such as Llama 2, locally. Ollama bundles model weights, configuration, and data into a single package, defined by a Modelfile. It optimizes setup and configuration details, including GPU usage.

- [Ollama + AutoGen instruction](https://ollama.ai/blog/openai-compatibility)
"""
logger.info("# Ollama")

logger.info("\n\n[DONE]", bright=True)