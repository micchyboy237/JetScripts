from IPython.display import display
from PIL import Image
from jet.logger import CustomLogger
from llama_index import SimpleDirectoryReader
from llama_index.multi_modal_llms.openai import MLXMultiModal
from llama_index.tools.openai.image_generation import MLXImageGenerationToolSpec
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Multi-Modal MLX Image Generation + GPT-4V
"""
logger.info("# Multi-Modal MLX Image Generation + GPT-4V")




image_generation_tool = MLXImageGenerationToolSpec(
#     api_key=os.environ["OPENAI_API_KEY"]
)

image_path = image_generation_tool.image_generation(
    "A pink and blue llama in a black background"
)




image_documents = SimpleDirectoryReader("../../../img_cache").load_data()

openai_mm_llm = MLXMultiModal(
    model="qwen3-1.7b-4bit",
#     api_key=os.environ["OPENAI_API_KEY"],
    max_new_tokens=300,
)

response = openai_mm_llm.complete(
    prompt="Describe the images as an alternative text", image_documents=image_documents
)

logger.debug(response)



img = Image.open(image_path)

display(img)

logger.info("\n\n[DONE]", bright=True)