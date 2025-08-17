import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
import * as readline from 'readline'
import os
import shutil
import { MLX } from 'openai'
import { Memory } from 'mem0ai/oss'


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: AI Companion in Node.js
---

You can create a personalised AI Companion using Mem0. This guide will walk you through the necessary steps and provide the complete code to get you started.

## Overview

The Personalized AI Companion leverages Mem0 to retain information across interactions, enabling a tailored learning experience. It creates memories for each user interaction and integrates with MLX's GPT models to provide detailed and context-aware responses to user queries.

## Setup

Before you begin, ensure you have Node.js installed and create a new project. Install the required dependencies using npm:
"""
logger.info("## Overview")

npm install openai mem0ai

"""
## Full Code Example

Below is the complete code to create and interact with an AI Companion using Mem0:
"""
logger.info("## Full Code Example")


openaiClient = new MLX()
memory = new Memory()

async function chatWithMemories(message, userId = "default_user") {
  async def run_async_code_710663d0():
      async def run_async_code_bb0456e9():
          relevantMemories = await memory.search(message, { userId: userId })
          return relevantMemories
      relevantMemories = asyncio.run(run_async_code_bb0456e9())
      logger.success(format_json(relevantMemories))
      return relevantMemories
  relevantMemories = asyncio.run(run_async_code_710663d0())
  logger.success(format_json(relevantMemories))

  memoriesStr = relevantMemories.results
    .map(lambda x: entry => `- ${entry.memory}`)
    .join('\n')

  systemPrompt = `You are a helpful AI. Answer the question based on query and memories.
User Memories:
${memoriesStr}`

  messages = [
    { role: "system", content: systemPrompt },
    { role: "user", content: message }
  ]

  async def run_async_code_7b10d7e8():
      response = await openaiClient.chat.completions.create({
      return response
  response = asyncio.run(run_async_code_7b10d7e8())
  logger.success(format_json(response))
    model: "llama-3.2-3b-instruct",
    messages: messages
  })

  assistantResponse = response.choices[0].message.content || ""

  messages.push({ role: "assistant", content: assistantResponse })
  async def run_async_code_8bfcd183():
      await memory.add(messages, { userId: userId })
      return 
   = asyncio.run(run_async_code_8bfcd183())
  logger.success(format_json())

  return assistantResponse
}

async function main() {
  rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  })

  console.log("Chat with AI (type 'exit' to quit)")

  askQuestion = lambda : {
    return new Promiselambda (resolve: {
      rl.questionlambda "You: ", (input: {
        resolve(input.trim())
      })
    })
  }

  try {
    while (true) {
      async def run_async_code_5dab38ea():
          async def run_async_code_0620f35e():
              userInput = await askQuestion()
              return userInput
          userInput = asyncio.run(run_async_code_0620f35e())
          logger.success(format_json(userInput))
          return userInput
      userInput = asyncio.run(run_async_code_5dab38ea())
      logger.success(format_json(userInput))

      if (userInput.toLowerCase() === 'exit') {
        console.log("Goodbye!")
        rl.close()
        break
      }

      async def run_async_code_ae43a71b():
          async def run_async_code_4080c65e():
              response = await chatWithMemories(userInput, "sample_user")
              return response
          response = asyncio.run(run_async_code_4080c65e())
          logger.success(format_json(response))
          return response
      response = asyncio.run(run_async_code_ae43a71b())
      logger.success(format_json(response))
      console.log(`AI: ${response}`)
    }
  } catch (error) {
    console.error("An error occurred:", error)
    rl.close()
  }
}

main().catch(console.error)

"""
### Key Components

1. **Initialization**
   - The code initializes both MLX and Mem0 Memory clients
   - Uses Node.js's built-in readline module for command-line interaction

2. **Memory Management (chatWithMemories function)**
   - Retrieves relevant memories using Mem0's search functionality
   - Constructs a system prompt that includes past memories
   - Makes API calls to MLX for generating responses
   - Stores new interactions in memory

3. **Interactive Chat Interface (main function)**
   - Creates a command-line interface for user interaction
   - Handles user input and displays AI responses
   - Includes graceful exit functionality

### Environment Setup

Make sure to set up your environment variables:
"""
logger.info("### Key Components")

# export OPENAI_API_KEY=your_api_key

"""
### Conclusion

This implementation demonstrates how to create an AI Companion that maintains context across conversations using Mem0's memory capabilities. The system automatically stores and retrieves relevant information, creating a more personalized and context-aware interaction experience.

As users interact with the system, Mem0's memory system continuously learns and adapts, making future responses more relevant and personalized. This setup is ideal for creating long-term learning AI assistants that can maintain context and provide increasingly personalized responses over time.
"""
logger.info("### Conclusion")

logger.info("\n\n[DONE]", bright=True)