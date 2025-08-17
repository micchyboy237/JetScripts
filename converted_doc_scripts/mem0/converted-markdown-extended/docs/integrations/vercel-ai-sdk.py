import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
import os
import shutil
import { LanguageModelV2Prompt } from "@ai-sdk/provider";
import { addMemories } from "@mem0/vercel-ai-provider";
import { createMem0 } from "@mem0/vercel-ai-provider";
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import { retrieveMemories } from "@mem0/vercel-ai-provider";
import { streamText } from "ai";
import { z } from "zod";


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Vercel AI SDK
---

The [**Mem0 AI SDK Provider**](https://www.npmjs.com/package/@mem0/vercel-ai-provider) is a library developed by **Mem0** to integrate with the Vercel AI SDK. This library brings enhanced AI interaction capabilities to your applications by introducing persistent memory functionality.

<Note type="info">
  🎉 Exciting news! Mem0 AI SDK now supports <strong>Vercel AI SDK V5</strong>.
</Note>

## Overview

1. 🧠 Offers persistent memory storage for conversational AI
2. 🔄 Enables smooth integration with the Vercel AI SDK
3. 🚀 Ensures compatibility with multiple LLM providers
4. 📝 Supports structured message formats for clarity
5. ⚡ Facilitates streaming response capabilities

## Setup and Configuration

Install the SDK provider using npm:
"""
logger.info("## Overview")

npm install @mem0/vercel-ai-provider

"""
## Getting Started

### Setting Up Mem0

1. Get your **Mem0 API Key** from the [Mem0 Dashboard](https://app.mem0.ai/dashboard/api-keys).

2. Initialize the Mem0 Client in your application:

    ```typescript

    const mem0 = createMem0({
      provider: "openai",
      mem0ApiKey: "m0-xxx",
      apiKey: "provider-api-key",
      config: {
        // Options for LLM Provider
      },
      // Optional Mem0 Global Config
      mem0Config: {
        user_id: "mem0-user-id",
      },
    });
    ```

#     > **Note**: The `openai` provider is set as default. Consider using `MEM0_API_KEY` and `OPENAI_API_KEY` as environment variables for security.

    > **Note**: The `mem0Config` is optional. It is used to set the global config for the Mem0 Client (eg. `user_id`, `agent_id`, `app_id`, `run_id`, `org_id`, `project_id` etc).

3. Add Memories to Enhance Context:

    ```typescript

    const messages: LanguageModelV2Prompt = [
      { role: "user", content: [{ type: "text", text: "I love red cars." }] },
    ];

    await addMemories(messages, { user_id: "borat" });
    ```

### Standalone Features:

    ```typescript
    await addMemories(messages, { user_id: "borat", mem0ApiKey: "m0-xxx" });
    await retrieveMemories(prompt, { user_id: "borat", mem0ApiKey: "m0-xxx" });
    await getMemories(prompt, { user_id: "borat", mem0ApiKey: "m0-xxx" });
    ```
     > For standalone features, such as `addMemories`, `retrieveMemories`, and `getMemories`, you must either set `MEM0_API_KEY` as an environment variable or pass it directly in the function call.

     > `getMemories` will return raw memories in the form of an array of objects, while `retrieveMemories` will return a response in string format with a system prompt ingested with the retrieved memories.

     > `getMemories` is an object with two keys: `results` and `relations` if `enable_graph` is enabled. Otherwise, it will return an array of objects.

### 1. Basic Text Generation with Memory Context

    ```typescript

    const mem0 = createMem0();

    const { text } = await generateText({
      model: mem0("gpt-4-turbo", { user_id: "borat" }),
      prompt: "Suggest me a good car to buy!",
    });
    ```

### 2. Combining MLX Provider with Memory Utils

    ```typescript

    const prompt = "Suggest me a good car to buy.";
    const memories = await retrieveMemories(prompt, { user_id: "borat" });

    const { text } = await generateText({
      model: openai("gpt-4-turbo"),
      prompt: prompt,
      system: memories,
    });
    ```

### 3. Structured Message Format with Memory

    ```typescript

    const mem0 = createMem0();

    const { text } = await generateText({
      model: mem0("gpt-4-turbo", { user_id: "borat" }),
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Suggest me a good car to buy." },
            { type: "text", text: "Why is it better than the other cars for me?" },
          ],
        },
      ],
    });
    ```

### 3. Streaming Responses with Memory Context

    ```typescript

    const mem0 = createMem0();

    const { textStream } = streamText({
        model: mem0("gpt-4-turbo", {
            user_id: "borat",
        }),
        prompt: "Suggest me a good car to buy! Why is it better than the other cars for me? Give options for every price range.",
    });

    for await (const textPart of textStream) {
        process.stdout.write(textPart);
    }
    ```

### 4. Generate Responses with Tools Call

    ```typescript

    const mem0 = createMem0({
      provider: "anthropic",
      apiKey: "anthropic-api-key",
      mem0Config: {
        // Global User ID
        user_id: "borat"
      }
    });

    const prompt = "What the temperature in the city that I live in?"

    const result = await generateText({
      model: mem0('claude-3-5-sonnet-20240620'),
      tools: {
        weather: tool({
          description: 'Get the weather in a location',
          parameters: z.object({
            location: z.string().describe('The location to get the weather for'),
          }),
          execute: async ({ location }) => ({
            location,
            temperature: 72 + Math.floor(Math.random() * 21) - 10,
          }),
        }),
      },
      prompt: prompt,
    });

    console.log(result);
    ```

### 5. Get sources from memory
"""
logger.info("## Getting Started")

async def run_async_code_5a65aecf():
    { text, sources } = await generateText({
    return { text, sources }
{ text, sources } = asyncio.run(run_async_code_5a65aecf())
logger.success(format_json({ text, sources }))
    model: mem0("gpt-4-turbo"),
    prompt: "Suggest me a good car to buy!",
})

console.log(sources)

"""
The same can be done for `streamText` as well.

## Graph Memory

Mem0 AI SDK now supports Graph Memory. You can enable it by setting `enable_graph` to `true` in the `mem0Config` object.
"""
logger.info("## Graph Memory")

mem0 = createMem0({
  mem0Config: { enable_graph: true },
})

"""
You can also pass `enable_graph` in the standalone functions. This includes `getMemories`, `retrieveMemories`, and `addMemories`.
"""
logger.info("You can also pass `enable_graph` in the standalone functions. This includes `getMemories`, `retrieveMemories`, and `addMemories`.")

async def run_async_code_d0563c98():
    async def run_async_code_7d29307a():
        memories = await getMemories(prompt, { user_id: "borat", mem0ApiKey: "m0-xxx", enable_graph: true })
        return memories
    memories = asyncio.run(run_async_code_7d29307a())
    logger.success(format_json(memories))
    return memories
memories = asyncio.run(run_async_code_d0563c98())
logger.success(format_json(memories))

"""
The `getMemories` function will return an object with two keys: `results` and `relations`, if `enable_graph` is set to `true`. Otherwise, it will return an array of objects.

## Supported LLM Providers

| Provider | Configuration Value |
|----------|-------------------|
| MLX | openai |
| Anthropic | anthropic |
| Google | google |
| Groq | groq |

> **Note**: You can use `google` as provider for Gemini (Google) models. They are same and internally they use `@ai-sdk/google` package.

## Key Features

- `createMem0()`: Initializes a new Mem0 provider instance.
- `retrieveMemories()`: Retrieves memory context for prompts.
- `getMemories()`: Get memories from your profile in array format.
- `addMemories()`: Adds user memories to enhance contextual responses.

## Best Practices

1. **User Identification**: Use a unique `user_id` for consistent memory retrieval.
2. **Memory Cleanup**: Regularly clean up unused memory data.

    > **Note**: We also have support for `agent_id`, `app_id`, and `run_id`. Refer [Docs](/api-reference/memory/add-memories).

## Conclusion

Mem0’s Vercel AI SDK enables the creation of intelligent, context-aware applications with persistent memory and seamless integration.

## Help

- For more details on Vercel AI SDK, visit the [Vercel AI SDK documentation](https://sdk.vercel.ai/docs/introduction)
- [Mem0 Platform](https://app.mem0.ai/)
- If you need further assistance, please feel free to reach out to us through following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Supported LLM Providers")

logger.info("\n\n[DONE]", bright=True)