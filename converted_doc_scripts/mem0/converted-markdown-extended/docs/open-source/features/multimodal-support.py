import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from mem0 import Memory
from mem0.exceptions import InvalidImageError, FileSizeError
import base64
import fs from 'fs'
import os
import shutil
import { Memory } from 'mem0ai'


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Multimodal Support
description: Integrate images into your interactions with Mem0
icon: "image"
iconType: "solid"
---

Mem0 extends its capabilities beyond text by supporting multimodal data. With this feature, you can seamlessly integrate images into your interactions—allowing Mem0 to extract relevant information and context from visual content.

## How It Works

When you submit an image, Mem0:
1. **Processes the visual content** using advanced vision models
2. **Extracts textual information** and relevant details from the image
3. **Stores the extracted information** as searchable memories
4. **Maintains context** between visual and textual interactions

This enables more comprehensive understanding of user interactions that include both text and visual elements.

<CodeGroup>
"""
logger.info("## How It Works")


client = Memory()

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

## Supported Image Formats

Mem0 supports common image formats:
- **JPEG/JPG** - Standard photos and images
- **PNG** - Images with transparency support  
- **WebP** - Modern web-optimized format
- **GIF** - Animated and static graphics

## Local Files vs URLs

### Using Image URLs
Images can be referenced via publicly accessible URLs:
"""
logger.info("## Supported Image Formats")

content = {
    "type": "image_url",
    "image_url": {
        "url": "https://example.com/my-image.jpg"
    }
}

"""
### Using Local Files
For local images, convert them to base64 format:

<CodeGroup>
"""
logger.info("### Using Local Files")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

client = Memory()

base64_image = encode_image("path/to/your/image.jpg")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What's in this image?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    }
]

client.add(messages, user_id="alice")

"""

"""


function encodeImage(imagePath) {
    imageBuffer = fs.readFileSync(imagePath)
    return imageBuffer.toString('base64')
}

client = new Memory()

base64Image = encodeImage("path/to/your/image.jpg")

messages = [
    {
        role: "user",
        content: [
            {
                type: "text",
                text: "What's in this image?"
            },
            {
                type: "image_url",
                image_url: {
                    url: `data:image/jpeg;base64,${base64Image}`
                }
            }
        ]
    }
]

async def run_async_code_0b3e9bea():
    await client.add(messages, { user_id: "alice" })
    return 
 = asyncio.run(run_async_code_0b3e9bea())
logger.success(format_json())

"""
</CodeGroup>

## Advanced Examples

### Restaurant Menu Analysis
"""
logger.info("## Advanced Examples")


client = Memory()

messages = [
    {
        "role": "user",
        "content": "I'm looking at this restaurant menu. Help me remember my preferences."
    },
    {
        "role": "user",
        "content": {
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/restaurant-menu.jpg"
            }
        }
    },
    {
        "role": "user",
        "content": "I'm allergic to peanuts and prefer vegetarian options."
    }
]

result = client.add(messages, user_id="user123")
logger.debug(result)

"""
### Document Analysis
"""
logger.info("### Document Analysis")

messages = [
    {
        "role": "user",
        "content": "Store this receipt information for my expense tracking."
    },
    {
        "role": "user",
        "content": {
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/receipt.jpg"
            }
        }
    }
]

client.add(messages, user_id="user123")

"""
## File Size and Performance Considerations

### Image Size Limits
- **Maximum file size**: 20MB per image
- **Recommended size**: Under 5MB for optimal performance
- **Resolution**: Images are automatically resized if needed

### Performance Tips
1. **Compress large images** before sending to reduce processing time
2. **Use appropriate formats** - JPEG for photos, PNG for graphics with text
3. **Batch processing** - Send multiple images in separate requests for better reliability

## Error Handling

Handle common errors when working with images:

<CodeGroup>
"""
logger.info("## File Size and Performance Considerations")


client = Memory()

try:
    messages = [{
        "role": "user",
        "content": {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.jpg"}
        }
    }]

    result = client.add(messages, user_id="user123")
    logger.debug("Image processed successfully")

except InvalidImageError:
    logger.debug("Invalid image format or corrupted file")
except FileSizeError:
    logger.debug("Image file too large")
except Exception as e:
    logger.debug(f"Unexpected error: {e}")

"""

"""


client = new Memory()

try {
    messages = [{
        role: "user",
        content: {
            type: "image_url",
            image_url: { url: "https://example.com/image.jpg" }
        }
    }]

    async def run_async_code_6e95a98d():
        async def run_async_code_22d9e641():
            result = await client.add(messages, { user_id: "user123" })
            return result
        result = asyncio.run(run_async_code_22d9e641())
        logger.success(format_json(result))
        return result
    result = asyncio.run(run_async_code_6e95a98d())
    logger.success(format_json(result))
    console.log("Image processed successfully")

} catch (error) {
    if (error.type === 'invalid_image') {
        console.log("Invalid image format or corrupted file")
    } else if (error.type === 'file_size_exceeded') {
        console.log("Image file too large")
    } else {
        console.log(`Unexpected error: ${error.message}`)
    }
}

"""
</CodeGroup>

## Best Practices

### Image Selection
- **Use high-quality images** with clear, readable text and details
- **Ensure good lighting** in photos for better text extraction
- **Avoid heavily stylized fonts** that may be difficult to read

### Memory Context
- **Provide context** about what information you want extracted
- **Combine with text** to give Mem0 better understanding of the image's purpose
- **Be specific** about what aspects of the image are important

### Privacy and Security
- **Avoid sensitive information** in images (SSN, passwords, private data)
- **Use secure image hosting** for URLs to prevent unauthorized access
- **Consider local processing** for highly sensitive visual content

Using these methods, you can seamlessly incorporate various visual content types into your interactions, further enhancing Mem0's multimodal capabilities for more comprehensive memory management.

If you have any questions, please feel free to reach out to us using one of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Best Practices")

logger.info("\n\n[DONE]", bright=True)