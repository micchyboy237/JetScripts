

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
myst:
  html_meta:
    "description lang=en": |
      User Guide for AutoGen Extensions, a framework for building multi-agent applications with AI agents.
---

# Installation

First-part maintained extensions are available in the `autogen-ext` package.
"""
logger.info("# Installation")

pip install "autogen-ext"

"""
Extras:

- `langchain` needed for {py:class}`~autogen_ext.tools.langchain.LangChainToolAdapter`
- `azure` needed for {py:class}`~autogen_ext.code_executors.azure.ACADynamicSessionsCodeExecutor`
- `docker` needed for {py:class}`~autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor`
- `openai` needed for {py:class}`~autogen_ext.models.openai.OllamaChatCompletionClient`
"""
logger.info("Extras:")

logger.info("\n\n[DONE]", bright=True)