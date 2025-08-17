from jet.logger import CustomLogger
from mem0 import Memory
import os
import shutil
import { Memory } from "mem0ai/oss";


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Features
description: 'Graph Memory features'
icon: "list-check"
iconType: "solid"
---

Graph Memory is a powerful feature that allows users to create and utilize complex relationships between pieces of information.

## Graph Memory supports the following features:

### Using Custom Prompts

Users can specify a custom prompt that will be used to extract specific entities from the given input text.
This allows for more targeted and relevant information extraction based on the user's needs.
Here's an example of how to specify a custom prompt:

<CodeGroup>
    ```python Python

    config = {
        "graph_store": {
            "provider": "neo4j",
            "config": {
                "url": "neo4j+s://xxx",
                "username": "neo4j",
                "password": "xxx"
            },
            "custom_prompt": "Please only extract entities containing sports related relationships and nothing else.",
        }
    }

    m = Memory.from_config(config_dict=config)
    ```

    ```typescript TypeScript

    const config = {
        graphStore: {
            provider: "neo4j",
            config: {
                url: "neo4j+s://xxx",
                username: "neo4j",
                password: "xxx",
            },
            customPrompt: "Please only extract entities containing sports related relationships and nothing else.",
        }
    }

    const memory = new Memory(config);
    ```
</CodeGroup>

If you want to use a managed version of Mem0, please check out [Mem0](https://mem0.dev/pd). If you have any questions, please feel free to reach out to us using one of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Graph Memory supports the following features:")

logger.info("\n\n[DONE]", bright=True)