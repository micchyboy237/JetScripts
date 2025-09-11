from google.colab.patches import cv2_imshow  # for image display
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.tools.ollama_dalle_image_generation import (
OllamaDALLEImageGenerationTool,
)
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import create_react_agent
from skimage import io
import cv2
import google.colab
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
# Dall-E Image Generator

>[Ollama Dall-E](https://ollama.com/dall-e-3) are text-to-image models developed by `Ollama` using deep learning methodologies to generate digital images from natural language descriptions, called "prompts".

This notebook shows how you can generate images from a prompt synthesized using an Ollama LLM. The images are generated using `Dall-E`, which uses the same Ollama API key as the LLM.
"""
logger.info("# Dall-E Image Generator")

# %pip install --upgrade --quiet  opencv-python scikit-image langchain-community



# os.environ["OPENAI_API_KEY"] = "insertapikey"

"""
## Run as a chain
"""
logger.info("## Run as a chain")


llm = Ollama(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["image_desc"],
    template="Generate a detailed prompt to generate an image based on the following description: {image_desc}",
)
chain = LLMChain(llm=llm, prompt=prompt)

image_url = DallEAPIWrapper().run(chain.run("halloween night at a haunted museum"))

image_url

try:

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:

    image = io.imread(image_url)
    cv2_imshow(image)
else:

    image = io.imread(image_url)
    cv2.imshow("image", image)
    cv2.waitKey(0)  # wait for a keyboard input
    cv2.destroyAllWindows()

"""
## Run as a tool with an agent
"""
logger.info("## Run as a tool with an agent")


llm = ChatOllama(model="llama3.2")
api_wrapper = DallEAPIWrapper()
dalle_tool = OllamaDALLEImageGenerationTool(api_wrapper=api_wrapper)

tools = [dalle_tool]

agent = create_react_agent(llm, tools, debug=True)

prompt = "Create an image of a halloween night at a haunted museum"

messages = [
    {"role": "user", "content": prompt}
]

response = agent.invoke({"messages": messages})

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)