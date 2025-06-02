from jet.logger import CustomLogger
from mlx_vlm import load, apply_chat_template, generate
from mlx_vlm.utils import load_image
from mlx_vlm.utils import process_image
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Multi-Image Generation

In this example, you will learn how to generate text from multiple images using the supported models: `Qwen2-VL`, `Pixtral` and `llava-interleaved`.

Multi-image generation allows you to pass a list of images to the model and generate text conditioned on all the images.
"""
logger.info("# Multi-Image Generation")


images = ["images/cats.jpg", "images/desktop_setup.png"]

messages = [
    {"role": "user", "content": "Describe what you see in the images."}
]

"""
## Qwen2-VL
"""
logger.info("## Qwen2-VL")

qwen_vl_model, qwen_vl_processor = load("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
qwen_vl_config = qwen_vl_model.config

prompt = apply_chat_template(qwen_vl_processor, qwen_vl_config, messages, num_images=len(images))

qwen_vl_output = generate(
    qwen_vl_model,
    qwen_vl_processor,
    prompt,
    images,
    max_tokens=1000,
    temperature=0.7,
    verbose=True
)

"""
## Pixtral
"""
logger.info("## Pixtral")

pixtral_model, pixtral_processor = load("mlx-community/pixtral-12b-4bit")
pixtral_config = pixtral_model.config

prompt = apply_chat_template(pixtral_processor, pixtral_config, messages, num_images=len(images))

resized_images = [process_image(load_image(image), (560, 560), None) for image in images]

pixtral_output = generate(
    pixtral_model,
    pixtral_processor,
    prompt,
    resized_images,
    max_tokens=1000,
    temperature=0.7,
    verbose=True
)

"""
## Llava-Interleaved
"""
logger.info("## Llava-Interleaved")

llava_model, llava_processor = load("mlx-community/llava-interleave-qwen-0.5b-bf16")
llava_config = llava_model.config

prompt = apply_chat_template(llava_processor, llava_config, messages, num_images=len(images))

llava_output = generate(
    llava_model,
    llava_processor,
    prompt,
    images,
    max_tokens=1000,
    temperature=0.7,
    verbose=True
)

logger.info("\n\n[DONE]", bright=True)