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
﻿# Release Notes for AutoGen.Net v0.2.1 🚀

## New Features 🌟
- **Support for OpenAi o1-preview** : Added support for Ollama o1-preview model ([#3522](https://github.com/microsoft/autogen/issues/3522))

## Example 📚
- **Ollama o1-preview**: [Connect_To_Ollama_o1_preview](https://github.com/microsoft/autogen/blob/main/dotnet/samples/AutoGen.Ollama.Sample/Connect_To_Ollama_o1_preview.cs)
"""
logger.info("## New Features 🌟")

logger.info("\n\n[DONE]", bright=True)