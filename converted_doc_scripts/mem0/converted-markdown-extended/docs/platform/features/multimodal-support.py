import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from mem0 import MemoryClient
import MemoryClient from "mem0ai"
import base64
import fs from 'fs'
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
title: Multimodal Support
description: Integrate images and documents into your interactions with Mem0
icon: "image"
iconType: "solid"
---

Mem0 extends its capabilities beyond text by supporting multimodal data, including images and documents. With this feature, users can seamlessly integrate visual and document content into their interactions—allowing Mem0 to extract relevant information from various media types and enrich the memory system.

## How It Works

When a user submits an image or document, Mem0 processes it to extract textual information and other pertinent details. These details are then added to the user's memory, enhancing the system's ability to understand and recall multimodal inputs.

<CodeGroup>
"""
logger.info("## How It Works")


os.environ["MEM0_API_KEY"] = "your-api-key"

client = MemoryClient()

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


client = new MemoryClient()

messages = [
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

async def run_async_code_0b3e9bea():
    await client.add(messages, { user_id: "alice" })
    return 
 = asyncio.run(run_async_code_0b3e9bea())
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

## Supported Media Types

Mem0 currently supports the following media types:

1. **Images** - JPG, PNG, and other common image formats
2. **Documents** - MDX, TXT, and PDF files

## Integration Methods

### 1. Images

#### Using an Image URL (Recommended)

You can include an image by providing its direct URL. This method is simple and efficient for online images.
"""
logger.info("## Supported Media Types")

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
client.add([image_message], user_id="alice")

"""
#### Using Base64 Image Encoding for Local Files

For local images—or when embedding the image directly is preferable—you can use a Base64-encoded string.

<CodeGroup>
"""
logger.info("#### Using Base64 Image Encoding for Local Files")


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
client.add([image_message], user_id="alice")

"""

"""


imagePath = 'path/to/your/image.jpg'

base64Image = fs.readFileSync(imagePath, { encoding: 'base64' })

imageMessage = {
    role: "user",
    content: {
        type: "image_url",
        image_url: {
            url: `data:image/jpeg;base64,${base64Image}`
        }
    }
}

async def run_async_code_ffc3bd0e():
    await client.add([imageMessage], { user_id: "alice" })
    return 
 = asyncio.run(run_async_code_ffc3bd0e())
logger.success(format_json())

"""
</CodeGroup>

### 2. Text Documents (MDX/TXT)

Mem0 supports both online and local text documents in MDX or TXT format.

#### Using a Document URL
"""
logger.info("### 2. Text Documents (MDX/TXT)")

document_url = "https://www.w3.org/TR/2003/REC-PNG-20031110/iso_8859-1.txt"

document_message = {
    "role": "user",
    "content": {
        "type": "mdx_url",
        "mdx_url": {
            "url": document_url
        }
    }
}
client.add([document_message], user_id="alice")

"""
#### Using Base64 Encoding for Local Documents
"""
logger.info("#### Using Base64 Encoding for Local Documents")


document_path = "path/to/your/document.txt"

def file_to_base64(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode('utf-8')

base64_document = file_to_base64(document_path)

document_message = {
    "role": "user",
    "content": {
        "type": "mdx_url",
        "mdx_url": {
            "url": base64_document
        }
    }
}
client.add([document_message], user_id="alice")

"""
### 3. PDF Documents

Mem0 supports PDF documents via URL.
"""
logger.info("### 3. PDF Documents")

pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

pdf_message = {
    "role": "user",
    "content": {
        "type": "pdf_url",
        "pdf_url": {
            "url": pdf_url
        }
    }
}
client.add([pdf_message], user_id="alice")

"""
## Complete Example with Multiple File Types

Here's a comprehensive example showing how to work with different file types:
"""
logger.info("## Complete Example with Multiple File Types")


client = MemoryClient()

def file_to_base64(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode('utf-8')

image_message = {
    "role": "user",
    "content": {
        "type": "image_url",
        "image_url": {
            "url": "https://example.com/sample-image.jpg"
        }
    }
}

text_message = {
    "role": "user",
    "content": {
        "type": "mdx_url",
        "mdx_url": {
            "url": "https://www.w3.org/TR/2003/REC-PNG-20031110/iso_8859-1.txt"
        }
    }
}

pdf_message = {
    "role": "user",
    "content": {
        "type": "pdf_url",
        "pdf_url": {
            "url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        }
    }
}

client.add([image_message], user_id="alice")
client.add([text_message], user_id="alice")
client.add([pdf_message], user_id="alice")

"""
Using these methods, you can seamlessly incorporate various media types into your interactions, further enhancing Mem0's multimodal capabilities.

If you have any questions, please feel free to reach out to us using one of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("Using these methods, you can seamlessly incorporate various media types into your interactions, further enhancing Mem0's multimodal capabilities.")

logger.info("\n\n[DONE]", bright=True)