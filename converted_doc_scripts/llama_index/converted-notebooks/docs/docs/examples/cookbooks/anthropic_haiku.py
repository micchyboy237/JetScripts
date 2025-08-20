from PIL import Image
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
import matplotlib.pyplot as plt
import os
import random
import shutil
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Anthropic Haiku Cookbook

Anthropic has released [Claude 3 Haiku](https://www.anthropic.com/news/claude-3-haiku). This notebook provides you to get a quick start with using the Haiku model. It helps you explore the capabilities of model on text and vision tasks.

#### Installation
"""
logger.info("# Anthropic Haiku Cookbook")

# !pip install llama-index
# !pip install llama-index-llms-anthropic
# !pip install llama-index-multi-modal-llms-anthropic


"""
#### Set API keys
"""
logger.info("#### Set API keys")


# os.environ["ANTHROPIC_API_KEY"] = "YOUR ANTHROPIC API KEY"

"""
### Using Model for Chat/ Completion
"""
logger.info("### Using Model for Chat/ Completion")

llm = Anthropic(model="claude-3-haiku-20240307")

response = llm.complete("LlamaIndex is ")
logger.debug(response)

"""
### Using Model for Multi-Modal

##### Download image
"""
logger.info("### Using Model for Multi-Modal")

# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/images/prometheus_paper_card.png' -O 'prometheus_paper_card.png'


img = Image.open("prometheus_paper_card.png")
plt.imshow(img)

"""
#### Load the image
"""
logger.info("#### Load the image")


image_documents = SimpleDirectoryReader(
    input_files=["prometheus_paper_card.png"]
).load_data()

anthropic_mm_llm = AnthropicMultiModal(
    model="claude-3-haiku-20240307", max_tokens=300
)

"""
#### Test query on image
"""
logger.info("#### Test query on image")

response = anthropic_mm_llm.complete(
    prompt="Describe the images as an alternative text",
    image_documents=image_documents,
)

logger.debug(response)

"""
#### Let's compare speed of the responses from different models

We will randomly generate 10 prompts and check the average response time.

##### Generate random 10 prompts
"""
logger.info("#### Let's compare speed of the responses from different models")


subjects = ["a cat", "an astronaut", "a teacher", "a robot", "a pirate"]
actions = [
    "is exploring a mysterious cave",
    "finds a hidden treasure",
    "solves a complex puzzle",
    "invents a new gadget",
    "discovers a new planet",
]

prompts = []
for _ in range(10):
    subject = random.choice(subjects)
    action = random.choice(actions)
    prompt = f"{subject} {action}"
    prompts.append(prompt)



def average_response_time(model, prompts):
    total_time_taken = 0
    llm = Anthropic(model=model, max_tokens=300)
    for prompt in prompts:
        start_time = time.time()
        _ = llm.complete(prompt)
        end_time = time.time()
        total_time_taken = total_time_taken + end_time - start_time

    return total_time_taken / len(prompts)

haiku_avg_response_time = average_response_time(
    "claude-3-haiku-20240307", prompts
)

opus_avg_response_time = average_response_time(
    "claude-3-opus-20240229", prompts
)

sonnet_avg_response_time = average_response_time(
    "claude-3-sonnet-20240229", prompts
)

logger.debug(f"Avg. time taken by Haiku model: {haiku_avg_response_time} seconds")
logger.debug(f"Avg. time taken by Opus model: {opus_avg_response_time} seconds")
logger.debug(f"Avg. time taken by Sonnet model: {sonnet_avg_response_time} seconds")

logger.info("\n\n[DONE]", bright=True)