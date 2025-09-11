from PIL import Image
from jet.logger import logger
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_vertexai import VertexAIImageCaptioning
from langchain_google_vertexai import VertexAIVisualQnAChat
from langchain_google_vertexai.vision_models import (
VertexAIImageEditorChat,
VertexAIImageGeneratorChat,
)
from langchain_google_vertexai.vision_models import VertexAIImageGeneratorChat
import base64
import io
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Google Imagen

>[Imagen on Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/image/overview) brings Google's state of the art image generative AI capabilities to application developers. With Imagen on Vertex AI, application developers can build next-generation AI products that transform their user's imagination into high quality visual assets using AI generation, in seconds.

With Imagen on Langchain , You can do the following tasks

- [VertexAIImageGeneratorChat](#image-generation) : Generate novel images using only a text prompt (text-to-image AI generation).
- [VertexAIImageEditorChat](#image-editing) : Edit an entire uploaded or generated image with a text prompt.
- [VertexAIImageCaptioning](#image-captioning) : Get text descriptions of images with visual captioning.
- [VertexAIVisualQnAChat](#visual-question-answering-vqa) : Get answers to a question about an image with Visual Question Answering (VQA).
    * NOTE : Currently we support only single-turn chat for Visual QnA (VQA)

## Image Generation
Generate novel images using only a text prompt (text-to-image AI generation)
"""
logger.info("# Google Imagen")


generator = VertexAIImageGeneratorChat()

messages = [HumanMessage(content=["a cat at the beach"])]
response = generator.invoke(messages)

generated_image = response.content[0]



img_base64 = generated_image["image_url"]["url"].split(",")[-1]

img = Image.open(io.BytesIO(base64.decodebytes(bytes(img_base64, "utf-8"))))

img

"""
## Image Editing
Edit an entire uploaded or generated image with a text prompt.

### Edit Generated Image
"""
logger.info("## Image Editing")


generator = VertexAIImageGeneratorChat()

messages = [HumanMessage(content=["a cat at the beach"])]

response = generator.invoke(messages)

generated_image = response.content[0]

editor = VertexAIImageEditorChat()

messages = [HumanMessage(content=[generated_image, "a dog at the beach "])]

editor_response = editor.invoke(messages)



edited_img_base64 = editor_response.content[0]["image_url"]["url"].split(",")[-1]

edited_img = Image.open(
    io.BytesIO(base64.decodebytes(bytes(edited_img_base64, "utf-8")))
)

edited_img

"""
## Image Captioning
"""
logger.info("## Image Captioning")


model = VertexAIImageCaptioning()

"""
NOTE :  we're using generated image in [Image Generation Section](#image-generation)
"""
logger.info("NOTE :  we're using generated image in [Image Generation Section](#image-generation)")

img_base64 = generated_image["image_url"]["url"]
response = model.invoke(img_base64)
logger.debug(f"Generated Caption : {response}")

img = Image.open(
    io.BytesIO(base64.decodebytes(bytes(img_base64.split(",")[-1], "utf-8")))
)

img

"""
## Visual Question Answering (VQA)
"""
logger.info("## Visual Question Answering (VQA)")


model = VertexAIVisualQnAChat()

"""
NOTE :  we're using generated image in [Image Generation Section](#image-generation)
"""
logger.info("NOTE :  we're using generated image in [Image Generation Section](#image-generation)")

question = "What animal is shown in the image?"
response = model.invoke(
    input=[
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": img_base64}},
                question,
            ]
        )
    ]
)

logger.debug(f"question : {question}\nanswer : {response.content}")

img = Image.open(
    io.BytesIO(base64.decodebytes(bytes(img_base64.split(",")[-1], "utf-8")))
)

img

logger.info("\n\n[DONE]", bright=True)