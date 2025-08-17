import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
import os
import shutil
import { Agent } from '@mastra/core/agent'
import { Mastra } from '@mastra/core/mastra'
import { Mem0Integration } from "@mastra/mem0"
import { createLogger } from '@mastra/core/logger'
import { createTool } from "@mastra/core"
import { mem0 } from "../integrations"
import { mem0Agent } from './agents'
import { mem0MemorizeTool, mem0RememberTool } from '../tools'
import { openai } from '@ai-sdk/openai'
import { z } from "zod"


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Mem0 with Mastra
---

In this example you'll learn how to use the Mem0 to add long-term memory capabilities to [Mastra's agent](https://mastra.ai/) via tool-use.
This memory integration can work alongside Mastra's [agent memory features](https://mastra.ai/docs/agents/01-agent-memory).

You can find the complete example code in the [Mastra repository](https://github.com/mastra-ai/mastra/tree/main/examples/memory-with-mem0).

## Overview

This guide will show you how to integrate Mem0 with Mastra to add long-term memory capabilities to your agents. We'll create tools that allow agents to save and retrieve memories using Mem0's API.

### Installation

1. **Install the Integration Package**

To install the Mem0 integration, run:
"""
logger.info("## Overview")

npm install @mastra/mem0

"""
2. **Add the Integration to Your Project**

Create a new file for your integrations and import the integration:
"""
logger.info("2. **Add the Integration to Your Project**")


export mem0 = new Mem0Integration({
  config: {
    apiKey: process.env.MEM0_API_KEY!,
    userId: "alice",
  },
})

"""
3. **Use the Integration in Tools or Workflows**

You can now use the integration when defining tools for your agents or in workflows.
"""
logger.info("3. **Use the Integration in Tools or Workflows**")


export mem0RememberTool = createTool({
  id: "Mem0-remember",
  description:
    "Remember your agent memories that you've previously saved using the Mem0-memorize tool.",
  inputSchema: z.object({
    question: z
      .string()
      .describe("Question used to look up the answer in saved memories."),
  }),
  outputSchema: z.object({
    answer: z.string().describe("Remembered answer"),
  }),
  execute: async lambda { context }: {
    console.log(`Searching memory "${context.question}"`)
    async def run_async_code_a2de9e7b():
        memory = await mem0.searchMemory(context.question)
        return memory
    memory = asyncio.run(run_async_code_a2de9e7b())
    logger.success(format_json(memory))
    console.log(`\nFound memory "${memory}"\n`)

    return {
      answer: memory,
    }
  },
})

export mem0MemorizeTool = createTool({
  id: "Mem0-memorize",
  description:
    "Save information to mem0 so you can remember it later using the Mem0-remember tool.",
  inputSchema: z.object({
    statement: z.string().describe("A statement to save into memory"),
  }),
  execute: async lambda { context }: {
    console.log(`\nCreating memory "${context.statement}"\n`)
    void mem0.createMemory(context.statement).thenlambda (: {
      console.log(`\nMemory "${context.statement}" saved.\n`)
    })
    return { success: true }
  },
})

"""
4. **Create a new agent**
"""
logger.info("4. **Create a new agent**")


export mem0Agent = new Agent({
  name: 'Mem0 Agent',
  instructions: `
    You are a helpful assistant that has the ability to memorize and remember facts using Mem0.
  `,
  model: openai('gpt-4o'),
  tools: { mem0RememberTool, mem0MemorizeTool },
})

"""
5. **Run the agent**
"""
logger.info("5. **Run the agent**")



export mastra = new Mastra({
  agents: { mem0Agent },
  logger: createLogger({
    name: 'Mastra',
    level: 'error',
  }),
})

"""
In the example above:
- We import the `@mastra/mem0` integration.
- We define two tools that uses the Mem0 API client to create new memories and recall previously saved memories.
- The tool accepts `question` as an input and returns the memory as a string.
"""
logger.info("In the example above:")

logger.info("\n\n[DONE]", bright=True)