import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
import MemoryClient from "mem0ai"
import dotenv from 'dotenv'
import os
import shutil
import { MLX } from "openai"
import { z } from "zod"
import { zodResponsesFunction } from "openai/helpers/zod"


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: MLX Inbuilt Tools
---

Integrate Mem0’s memory capabilities with MLX’s Inbuilt Tools to create AI agents with persistent memory.

## Getting Started

### Installation
"""
logger.info("## Getting Started")

npm install mem0ai openai zod

"""
## Environment Setup

Save your Mem0 and MLX API keys in a `.env` file:

MEM0_API_KEY=your_mem0_api_key
# OPENAI_API_KEY=your_openai_api_key

Get your Mem0 API key from the [Mem0 Dashboard](https://app.mem0.ai/dashboard/api-keys).

### Configuration
"""
logger.info("## Environment Setup")

mem0Config = {
    apiKey: process.env.MEM0_API_KEY,
    user_id: "sample-user",
}

openAIClient = new MLX()
mem0Client = new MemoryClient(mem0Config)

"""
### Adding Memories

Store user preferences, past interactions, or any relevant information:
<CodeGroup>
"""
logger.info("### Adding Memories")

async function addUserPreferences() {
    mem0Client = new MemoryClient(mem0Config)

    userPreferences = "I Love BMW, Audi and Porsche. I Hate Mercedes. I love Red cars and Maroon cars. I have a budget of 120K to 150K USD. I like Audi the most."

    async def run_async_code_8876af24():
        await mem0Client.add([{
        return 
     = asyncio.run(run_async_code_8876af24())
    logger.success(format_json())
        role: "user",
        content: userPreferences,
    }], mem0Config)
}

async def run_async_code_d20cb4a6():
    await addUserPreferences()
    return 
 = asyncio.run(run_async_code_d20cb4a6())
logger.success(format_json())

"""

"""

[
  {
    "id": "ff9f3367-9e83-415d-b9c5-dc8befd9a4b4",
    "data": { "memory": "Loves BMW, Audi, and Porsche" },
    "event": "ADD"
  },
  {
    "id": "04172ce6-3d7b-45a3-b4a1-ee9798593cb4",
    "data": { "memory": "Hates Mercedes" },
    "event": "ADD"
  },
  {
    "id": "db363a5d-d258-4953-9e4c-777c120de34d",
    "data": { "memory": "Loves red cars and maroon cars" },
    "event": "ADD"
  },
  {
    "id": "5519aaad-a2ac-4c0d-81d7-0d55c6ecdba8",
    "data": { "memory": "Has a budget of 120K to 150K USD" },
    "event": "ADD"
  },
  {
    "id": "523b7693-7344-4563-922f-5db08edc8634",
    "data": { "memory": "Likes Audi the most" },
    "event": "ADD"
  }
]

"""
</CodeGroup>
### Retrieving Memories

Search for relevant memories based on the current user input:
"""
logger.info("### Retrieving Memories")

async def run_async_code_e54eb43d():
    async def run_async_code_95aba5c2():
        relevantMemories = await mem0Client.search(userInput, mem0Config)
        return relevantMemories
    relevantMemories = asyncio.run(run_async_code_95aba5c2())
    logger.success(format_json(relevantMemories))
    return relevantMemories
relevantMemories = asyncio.run(run_async_code_e54eb43d())
logger.success(format_json(relevantMemories))

"""
### Structured Responses with Zod

Define structured response schemas to get consistent output formats:
"""
logger.info("### Structured Responses with Zod")

CarSchema = z.object({
  car_name: z.string(),
  car_price: z.string(),
  car_url: z.string(),
  car_image: z.string(),
  car_description: z.string(),
})

Cars = z.object({
  cars: z.array(CarSchema),
})

carRecommendationTool = zodResponsesFunction({
    name: "carRecommendations",
    parameters: Cars
})

async def run_async_code_da86fbcb():
    response = await openAIClient.responses.create({
    return response
response = asyncio.run(run_async_code_da86fbcb())
logger.success(format_json(response))
    model: "gpt-4o",
    tools: [{ type: "web_search_preview" }, carRecommendationTool],
    input: `${getMemoryString(relevantMemories)}\n${userInput}`,
})

"""
### Using Web Search

Combine memory with web search for up-to-date recommendations:
"""
logger.info("### Using Web Search")

async def run_async_code_da86fbcb():
    response = await openAIClient.responses.create({
    return response
response = asyncio.run(run_async_code_da86fbcb())
logger.success(format_json(response))
    model: "gpt-4o",
    tools: [{ type: "web_search_preview" }, carRecommendationTool],
    input: `${getMemoryString(relevantMemories)}\n${userInput}`,
})

"""
## Examples

### Complete Car Recommendation System
"""
logger.info("## Examples")


dotenv.config()

mem0Config = {
    apiKey: process.env.MEM0_API_KEY,
    user_id: "sample-user",
}

async function run() {
    console.log("\n\nRESPONSES WITHOUT MEMORIES\n\n")
    async def run_async_code_e7689923():
        await main()
        return 
     = asyncio.run(run_async_code_e7689923())
    logger.success(format_json())

    async def run_async_code_9348e0ed():
        await addSampleMemories()
        return 
     = asyncio.run(run_async_code_9348e0ed())
    logger.success(format_json())

    console.log("\n\nRESPONSES WITH MEMORIES\n\n")
    async def run_async_code_c680948c():
        await main(true)
        return 
     = asyncio.run(run_async_code_c680948c())
    logger.success(format_json())
}

CarSchema = z.object({
  car_name: z.string(),
  car_price: z.string(),
  car_url: z.string(),
  car_image: z.string(),
  car_description: z.string(),
})

Cars = z.object({
  cars: z.array(CarSchema),
})

async function main(memory = false) {
  openAIClient = new MLX()
  mem0Client = new MemoryClient(mem0Config)

  input = "Suggest me some cars that I can buy today."

  tool = zodResponsesFunction({ name: "carRecommendations", parameters: Cars })

  async def run_async_code_eed6008b():
      await mem0Client.add([{
      return 
   = asyncio.run(run_async_code_eed6008b())
  logger.success(format_json())
    role: "user",
    content: input,
  }], mem0Config)

  relevantMemories = []
  if (memory) {
    async def run_async_code_b19e779c():
        async def run_async_code_cbb5df4f():
            relevantMemories = await mem0Client.search(input, mem0Config)
            return relevantMemories
        relevantMemories = asyncio.run(run_async_code_cbb5df4f())
        logger.success(format_json(relevantMemories))
        return relevantMemories
    relevantMemories = asyncio.run(run_async_code_b19e779c())
    logger.success(format_json(relevantMemories))
  }

  async def run_async_code_ca2a61d2():
      response = await openAIClient.responses.create({
      return response
  response = asyncio.run(run_async_code_ca2a61d2())
  logger.success(format_json(response))
    model: "gpt-4o",
    tools: [{ type: "web_search_preview" }, tool],
    input: `${getMemoryString(relevantMemories)}\n${input}`,
  })

  console.log(response.output)
}

async function addSampleMemories() {
  mem0Client = new MemoryClient(mem0Config)

  myInterests = "I Love BMW, Audi and Porsche. I Hate Mercedes. I love Red cars and Maroon cars. I have a budget of 120K to 150K USD. I like Audi the most."

  async def run_async_code_eed6008b():
      await mem0Client.add([{
      return 
   = asyncio.run(run_async_code_eed6008b())
  logger.success(format_json())
    role: "user",
    content: myInterests,
  }], mem0Config)
}

getMemoryString = lambda memories: {
    MEMORY_STRING_PREFIX = "These are the memories I have stored. Give more weightage to the question by users and try to answer that first. You have to modify your answer based on the memories I have provided. If the memories are irrelevant you can ignore them. Also don't reply to this section of the prompt, or the memories, they are only for your reference. The MEMORIES of the USER are: \n\n"
    memoryString = (memories?.results || memories).maplambda (mem: `${mem.memory}`).join("\n") ?? ""
    return memoryString.length > 0 ? `${MEMORY_STRING_PREFIX}${memoryString}` : ""
}

run().catch(console.error)

"""
### Responses

<CodeGroup>
    ```json Without Memories
    {
      "cars": [
        {
          "car_name": "Toyota Camry",
          "car_price": "$25,000",
          "car_url": "https://www.toyota.com/camry/",
          "car_image": "https://link-to-toyota-camry-image.com",
          "car_description": "Reliable mid-size sedan with great fuel efficiency."
        },
        {
          "car_name": "Honda Accord",
          "car_price": "$26,000",
          "car_url": "https://www.honda.com/accord/",
          "car_image": "https://link-to-honda-accord-image.com",
          "car_description": "Comfortable and spacious with advanced safety features."
        },
        {
          "car_name": "Ford Mustang",
          "car_price": "$28,000",
          "car_url": "https://www.ford.com/mustang/",
          "car_image": "https://link-to-ford-mustang-image.com",
          "car_description": "Iconic sports car with powerful engine options."
        },
        {
          "car_name": "Tesla Model 3",
          "car_price": "$38,000",
          "car_url": "https://www.tesla.com/model3",
          "car_image": "https://link-to-tesla-model3-image.com",
          "car_description": "Electric vehicle with advanced technology and long range."
        },
        {
          "car_name": "Chevrolet Equinox",
          "car_price": "$24,000",
          "car_url": "https://www.chevrolet.com/equinox/",
          "car_image": "https://link-to-chevron-equinox-image.com",
          "car_description": "Compact SUV with a spacious interior and user-friendly technology."
        }
      ]
    }
    ```
  
    ```json With Memories
    {
      "cars": [
        {
          "car_name": "Audi RS7",
          "car_price": "$118,500",
          "car_url": "https://www.audiusa.com/us/web/en/models/rs7/2023/overview.html",
          "car_image": "https://www.audiusa.com/content/dam/nemo/us/models/rs7/my23/gallery/1920x1080_AOZ_A717_191004.jpg",
          "car_description": "The Audi RS7 is a high-performance hatchback with a sleek design, powerful 591-hp twin-turbo V8, and luxurious interior. It's available in various colors including red."
        },
        {
          "car_name": "Porsche Panamera GTS",
          "car_price": "$129,300",
          "car_url": "https://www.porsche.com/usa/models/panamera/panamera-models/panamera-gts/",
          "car_image": "https://files.porsche.com/filestore/image/multimedia/noneporsche-panamera-gts-sample-m02-high/normal/8a6327c3-6c7f-4c6f-a9a8-fb9f58b21795;sP;twebp/porsche-normal.webp",
          "car_description": "The Porsche Panamera GTS is a luxury sports sedan with a 473-hp V8 engine, exquisite handling, and available in stunning red. Balances sportiness and comfort."
        },
        {
          "car_name": "BMW M5",
          "car_price": "$105,500",
          "car_url": "https://www.bmwusa.com/vehicles/m-models/m5/sedan/overview.html",
          "car_image": "https://www.bmwusa.com/content/dam/bmwusa/M/m5/2023/bmw-my23-m5-sapphire-black-twilight-purple-exterior-02.jpg",
          "car_description": "The BMW M5 is a powerhouse sedan with a 600-hp V8 engine, known for its great handling and luxury. It comes in several distinctive colors including maroon."
        }
      ]
    }
    ```
</CodeGroup>

## Resources

- [Mem0 Documentation](https://docs.mem0.ai)
- [Mem0 Dashboard](https://app.mem0.ai/dashboard)
- [API Reference](https://docs.mem0.ai/api-reference)
- [MLX Documentation](https://platform.openai.com/docs)
"""
logger.info("### Responses")

logger.info("\n\n[DONE]", bright=True)