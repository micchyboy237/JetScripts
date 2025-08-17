import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from mem0 import Memory
import os
import shutil
import { Memory } from 'mem0ai/oss'


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Custom Fact Extraction Prompt
description: 'Enhance your product experience by adding custom fact extraction prompt tailored to your needs'
icon: "pencil"
iconType: "solid"
---

## Introduction to Custom Fact Extraction Prompt

Custom fact extraction prompt allow you to tailor the behavior of your Mem0 instance to specific use cases or domains. 
By defining it, you can control how information is extracted from the user's message.

To create an effective custom fact extraction prompt:
1. Be specific about the information to extract.
2. Provide few-shot examples to guide the LLM.
3. Ensure examples follow the format shown below.

Example of a custom fact extraction prompt:

<CodeGroup>
"""
logger.info("## Introduction to Custom Fact Extraction Prompt")

custom_fact_extraction_prompt = """
Please only extract entities containing customer support information, order details, and user information.
Here are some few shot examples:

Input: Hi.
Output: {{"facts" : []}}

Input: The weather is nice today.
Output: {{"facts" : []}}

Input: My order #12345 hasn't arrived yet.
Output: {{"facts" : ["Order #12345 not received"]}}

Input: I'm John Doe, and I'd like to return the shoes I bought last week.
Output: {{"facts" : ["Customer name: John Doe", "Wants to return shoes", "Purchase made last week"]}}

Input: I ordered a red shirt, size medium, but received a blue one instead.
Output: {{"facts" : ["Ordered red shirt, size medium", "Received blue shirt instead"]}}

Return the facts and customer information in a json format as shown above.
"""

"""

"""

customPrompt = `
Please only extract entities containing customer support information, order details, and user information.
Here are some few shot examples:

Input: Hi.
Output: {"facts" : []}

Input: The weather is nice today.
Output: {"facts" : []}

Input: My order #12345 hasn't arrived yet.
Output: {"facts" : ["Order #12345 not received"]}

Input: I am John Doe, and I would like to return the shoes I bought last week.
Output: {"facts" : ["Customer name: John Doe", "Wants to return shoes", "Purchase made last week"]}

Input: I ordered a red shirt, size medium, but received a blue one instead.
Output: {"facts" : ["Ordered red shirt, size medium", "Received blue shirt instead"]}

Return the facts and customer information in a json format as shown above.
`

"""
</CodeGroup>

Here we initialize the custom fact extraction prompt in the config:

<CodeGroup>
"""
logger.info("Here we initialize the custom fact extraction prompt in the config:")


config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o",
            "temperature": 0.2,
            "max_tokens": 2000,
        }
    },
    "custom_fact_extraction_prompt": custom_fact_extraction_prompt,
    "version": "v1.1"
}

m = Memory.from_config(config_dict=config, user_id="alice")

"""

"""


config = {
  version: 'v1.1',
  llm: {
    provider: 'openai',
    config: {
#       apiKey: process.env.OPENAI_API_KEY || '',
      model: 'gpt-4-turbo-preview',
      temperature: 0.2,
      maxTokens: 1500,
    },
  },
  customPrompt: customPrompt
}

memory = new Memory(config)

"""
</CodeGroup>

### Example 1

In this example, we are adding a memory of a user ordering a laptop. As seen in the output, the custom prompt is used to extract the relevant information from the user's message.

<CodeGroup>
"""
logger.info("### Example 1")

m.add("Yesterday, I ordered a laptop, the order id is 12345", user_id="alice")

"""

"""

async def run_async_code_9a8e7c9b():
    await memory.add('Yesterday, I ordered a laptop, the order id is 12345', { userId: "user123" })
    return 
 = asyncio.run(run_async_code_9a8e7c9b())
logger.success(format_json())

"""

"""

{
  "results": [
    {
      "memory": "Ordered a laptop",
      "event": "ADD"
    },
    {
      "memory": "Order ID: 12345",
      "event": "ADD"
    },
    {
      "memory": "Order placed yesterday",
      "event": "ADD"
    }
  ],
  "relations": []
}

"""
</CodeGroup>

### Example 2

In this example, we are adding a memory of a user liking to go on hikes. This add message is not specific to the use-case mentioned in the custom prompt. 
Hence, the memory is not added.

<CodeGroup>
"""
logger.info("### Example 2")

m.add("I like going to hikes", user_id="alice")

"""

"""

async def run_async_code_879f6e5a():
    await memory.add('I like going to hikes', { userId: "user123" })
    return 
 = asyncio.run(run_async_code_879f6e5a())
logger.success(format_json())

"""

"""

{
  "results": [],
  "relations": []
}

"""
</CodeGroup>

The custom fact extraction prompt will process both the user and assistant messages to extract relevant information according to the defined format.
"""
logger.info("The custom fact extraction prompt will process both the user and assistant messages to extract relevant information according to the defined format.")

logger.info("\n\n[DONE]", bright=True)