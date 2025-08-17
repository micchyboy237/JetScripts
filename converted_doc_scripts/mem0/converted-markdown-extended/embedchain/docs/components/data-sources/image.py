from embedchain import App
from embedchain.loaders.image import ImageLoader
from jet.logger import CustomLogger
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
title: "🖼️ Image"
---


To use an image as data source, just add `data_type` as `image` and pass in the path of the image (local or hosted).

We use [GPT4 Vision](https://platform.openai.com/docs/guides/vision) to generate meaning of the image using a custom prompt, and then use the generated text as the data source.

You would require an MLX API key with access to `gpt-4-vision-preview` model to use this feature.

### Without customization
"""
logger.info("### Without customization")


# os.environ["OPENAI_API_KEY"] = "sk-xxx"

app = App()
app.add("./Elon-Musk.webp", data_type="image")
response = app.query("Describe the man in the image.")
logger.debug(response)

"""
### Customization
"""
logger.info("### Customization")


image_loader = ImageLoader(
    max_tokens=100,
    api_key="sk-xxx",
    prompt="Is the person looking wealthy? Structure your thoughts around what you see in the image.",
)

app = App()
app.add("./Elon-Musk.webp", data_type="image", loader=image_loader)
response = app.query("Describe the man in the image.")
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)