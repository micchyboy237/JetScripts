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
---
title: Eliza OS Character
---

You can create a personalised Eliza OS Character using Mem0. This guide will walk you through the necessary steps and provide the complete code to get you started.

## Overview

ElizaOS is a powerful AI agent framework for autonomy & personality. It is a collection of tools that help you create a personalised AI agent.

## Setup
You can start by cloning the eliza-os repository:
"""
logger.info("## Overview")

git clone https://github.com/elizaOS/eliza.git

"""
Change the directory to the eliza-os repository:
"""
logger.info("Change the directory to the eliza-os repository:")

cd eliza

"""
Install the dependencies:
"""
logger.info("Install the dependencies:")

pnpm install

"""
Build the project:
"""
logger.info("Build the project:")

pnpm build

"""
## Setup ENVs

Create a `.env` file in the root of the project and add the following ( You can use the `.env.example` file as a reference):
"""
logger.info("## Setup ENVs")

MEM0_API_KEY= # Mem0 API Key ( Get from https://app.mem0.ai/dashboard/api-keys )
MEM0_USER_ID= # Default: eliza-os-user
MEM0_PROVIDER= # Default: openai
MEM0_PROVIDER_API_KEY= # API Key for the provider (openai, anthropic, etc.)
SMALL_MEM0_MODEL= # Default: llama-3.2-3b-instruct
MEDIUM_MEM0_MODEL= # Default: gpt-4o
LARGE_MEM0_MODEL= # Default: gpt-4o

"""
## Make the default character use Mem0

By default, there is a character called `eliza` that uses the `ollama` model. You can make this character use Mem0 by changing the config in the `agent/src/defaultCharacter.ts` file.
"""
logger.info("## Make the default character use Mem0")

modelProvider: ModelProviderName.MEM0,

"""
This will make the character use Mem0 to generate responses.

## Run the project
"""
logger.info("## Run the project")

pnpm start

"""
## Conclusion

You have now created a personalised Eliza OS Character using Mem0. You can now start interacting with the character by running the project and talking to the character.

This is a simple example of how to use Mem0 to create a personalised AI agent. You can use this as a starting point to create your own AI agent.
"""
logger.info("## Conclusion")

logger.info("\n\n[DONE]", bright=True)