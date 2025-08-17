import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from mem0 import Memory
import base64
import os
import shutil
import { Memory, Message } from "mem0ai/oss"


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Multimodal Support
icon: "image"
iconType: "solid"
---

Mem0 extends its capabilities beyond text by supporting multimodal data, including images. Users can seamlessly integrate images into their interactions, allowing Mem0 to extract pertinent information from visual content and enrich the memory system.

## How It Works

When a user provides an image, Mem0 processes the image to extract textual information and relevant details, which are then added to the user's memory. This feature enhances the system's ability to understand and remember details based on visual inputs.

<Note>
To enable multimodal support, you must set `enable_vision = True` in your configuration. The `vision_details` parameter can be set to "auto" (default), "low", or "high" to control the level of detail in image processing.
</Note>

<CodeGroup>
"""
logger.info("## How It Works")


config = {
    "llm": {
        "provider": "openai",
        "config": {
            "enable_vision": True,
            "vision_details": "high"
        }
    }
}

client = Memory.from_config(config=config)

messages = [
    {
        "role": "user",
        "content": "Hi, my name is Alice."
    },
    {
        "role": "assistant",
        "content": "Nice to meet you, Alice! What do you like to eat?"
    },
    {
        "role": "user",
        "content": {
            "type": "image_url",
            "image_url": {
                "url": "https://www.superhealthykids.com/wp-content/uploads/2021/10/best-veggie-pizza-featured-image-square-2.jpg"
            }
        }
    },
]

client.add(messages, user_id="alice")

"""

"""


client = new Memory()

messages: Message[] = [
    {
        role: "user",
        content: "Hi, my name is Alice."
    },
    {
        role: "assistant",
        content: "Nice to meet you, Alice! What do you like to eat?"
    },
    {
        role: "user",
        content: {
            type: "image_url",
            image_url: {
                url: "https://www.superhealthykids.com/wp-content/uploads/2021/10/best-veggie-pizza-featured-image-square-2.jpg"
            }
        }
    },
]

async def run_async_code_bcf65193():
    await client.add(messages, { userId: "alice" })
    return 
 = asyncio.run(run_async_code_bcf65193())
logger.success(format_json())

"""

"""

{
  "results": [
    {
      "memory": "Name is Alice",
      "event": "ADD",
      "id": "7ae113a3-3cb5-46e9-b6f7-486c36391847"
    },
    {
      "memory": "Likes large pizza with toppings including cherry tomatoes, black olives, green spinach, yellow bell peppers, diced ham, and sliced mushrooms",
      "event": "ADD",
      "id": "56545065-7dee-4acf-8bf2-a5b2535aabb3"
    }
  ]
}

"""
</CodeGroup>

## Image Integration Methods

Mem0 allows you to add images to user interactions through two primary methods: by providing an image URL or by using a Base64-encoded image. Below are examples demonstrating each approach.

## 1. Using an Image URL (Recommended)

You can include an image by passing its direct URL. This method is simple and efficient for online images.

<CodeGroup>
"""
logger.info("## Image Integration Methods")

image_url = "https://www.superhealthykids.com/wp-content/uploads/2021/10/best-veggie-pizza-featured-image-square-2.jpg"

image_message = {
    "role": "user",
    "content": {
        "type": "image_url",
        "image_url": {
            "url": image_url
        }
    }
}

"""

"""


client = new Memory()

imageUrl = "https://www.superhealthykids.com/wp-content/uploads/2021/10/best-veggie-pizza-featured-image-square-2.jpg"

imageMessage: Message = {
    role: "user",
    content: {
        type: "image_url",
        image_url: {
            url: imageUrl
        }
    }
}

async def run_async_code_3b5f05b6():
    await client.add([imageMessage], { userId: "alice" })
    return 
 = asyncio.run(run_async_code_3b5f05b6())
logger.success(format_json())

"""
</CodeGroup>

## 2. Using Base64 Image Encoding for Local Files

For local images or scenarios where embedding the image directly is preferable, you can use a Base64-encoded string.

<CodeGroup>
"""
logger.info("## 2. Using Base64 Image Encoding for Local Files")


image_path = "path/to/your/image.jpg"

with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

image_message = {
    "role": "user",
    "content": {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    }
}

"""

"""


client = new Memory()

imagePath = "path/to/your/image.jpg"

base64Image = fs.readFileSync(imagePath, { encoding: 'base64' })

imageMessage: Message = {
    role: "user",
    content: {
        type: "image_url",
        image_url: {
            url: `data:image/jpeg;base64,${base64Image}`
        }
    }
}

async def run_async_code_3b5f05b6():
    await client.add([imageMessage], { userId: "alice" })
    return 
 = asyncio.run(run_async_code_3b5f05b6())
logger.success(format_json())

"""
</CodeGroup>

## 3. MLX-Compatible Message Format

You can also use the MLX-compatible format to combine text and images in a single message:

<CodeGroup>
"""
logger.info("## 3. MLX-Compatible Message Format")


image_path = "path/to/your/image.jpg"

with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "What is in this image?",
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        },
    ],
}

client.add([message], user_id="alice")

"""

"""


client = new Memory()

imagePath = "path/to/your/image.jpg"

base64Image = fs.readFileSync(imagePath, { encoding: 'base64' })

message: Message = {
    role: "user",
    content: [
        {
            type: "text",
            text: "What is in this image?",
        },
        {
            type: "image_url",
            image_url: {
                url: `data:image/jpeg;base64,${base64Image}`
            }
        },
    ],
}

async def run_async_code_239fcce6():
    await client.add([message], { userId: "alice" })
    return 
 = asyncio.run(run_async_code_239fcce6())
logger.success(format_json())

"""
</CodeGroup>

This format allows you to combine text and images in a single message, making it easier to provide context along with visual content.

By utilizing these methods, you can effectively incorporate images into user interactions, enhancing the multimodal capabilities of your Mem0 instance.

If you have any questions, please feel free to reach out to us using one of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("This format allows you to combine text and images in a single message, making it easier to provide context along with visual content.")

logger.info("\n\n[DONE]", bright=True)