from IPython.display import Latex
from jet.logger import CustomLogger
from mlx_vlm import load, apply_chat_template, generate
from mlx_vlm.utils import load_image
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Text extraction

In this example, you will learn how to extract text from images/documents and format the output in different ways using MLX-VLM library and `Qwen2-VL` model.


## Structured outputs
- Image to markdown
- Image to json
- Image to latex
"""
logger.info("# Text extraction")


qwen_vl_model, qwen_vl_processor = load("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
qwen_vl_config = qwen_vl_model.config

"""
## Image to markdown
"""
logger.info("## Image to markdown")

image = load_image("images/paper.png")

messages = [
    {"role": "system", "content": """
    You are an expert at extracting text from images. Format your response in markdown.
    Format your response as follows:
    - Authors, Affiliation, Email
    Paragraph
    """},
    {"role": "user", "content": "Extract all the text from the image."}
]

prompt = apply_chat_template(qwen_vl_processor, qwen_vl_config, messages)

qwen_vl_output = generate(
    qwen_vl_model,
    qwen_vl_processor,
    prompt,
    image,
    max_tokens=1000,
    temperature=0.7,
    verbose=True,
)

logger.debug(qwen_vl_output)

image

"""
## Image to json
"""
logger.info("## Image to json")

image = load_image("images/graph.png")

messages = [
    {"role": "system", "content": "You are an expert at extracting text from images. Format your response in json."},
    {"role": "user", "content": "Extract the names, labels and y coordinates from the image."}
]

prompt = apply_chat_template(qwen_vl_processor, qwen_vl_config, messages)

qwen_vl_output = generate(
    qwen_vl_model,
    qwen_vl_processor,
    prompt,
    image,
    max_tokens=1000,
    temperature=0.7,
    verbose=True,
)

logger.debug(qwen_vl_output)

image

"""
## Image to latex
"""
logger.info("## Image to latex")

image = load_image("images/latex.png")

messages = [
    {"role": "user", "content": "Extract the text from the image and format it into latex."}
]

prompt = apply_chat_template(qwen_vl_processor, qwen_vl_config, messages)

qwen_vl_output = generate(
    qwen_vl_model,
    qwen_vl_processor,
    prompt,
    image,
    max_tokens=1000,
    temperature=0.7,
    verbose=True,
)

Latex(qwen_vl_output)

image

logger.info("\n\n[DONE]", bright=True)