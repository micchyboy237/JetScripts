from IPython.display import display
from PIL import Image
from jet.logger import CustomLogger
from llama_index.agent import ReActAgent
from llama_index.tools import FunctionTool
from llama_index.tools.openai.image_generation import OllamaFunctionCallingAdapterImageGenerationToolSpec
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# OllamaFunctionCalling Image Generation Agent (DALL-E-3)
"""
logger.info("# OllamaFunctionCalling Image Generation Agent (DALL-E-3)")







def show_image(filename: str) -> Image:
    """Display an image based on the filename"""
    img = Image.open(filename)
    return display(img)


image_generation_tool = OllamaFunctionCallingAdapterImageGenerationToolSpec(
#     api_key=os.environ["OPENAI_API_KEY"]
)
show_image_tool = FunctionTool.from_defaults(fn=show_image)

agent = ReActAgent.from_tools(
    [*image_generation_spec.to_tool_list(), show_image_tool], verbose=True
)

response = agent.query(
    "generate a hacker image with size 1024x1024, use the filename and show the image"
)

logger.info("\n\n[DONE]", bright=True)