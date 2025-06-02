from jet.logger import CustomLogger
from mlx_vlm import load, apply_chat_template, generate
from mlx_vlm.utils import load_image
from utils import parse_bbox, plot_image_with_bboxes
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Object Detection


In this example, you will learn how to perform object detection using the MLX-VLM (Vision Language Model) library. 

We will use two different models: `Qwen2-VL` and `Paligemma`, for detecting objects in images and generating bounding boxes.

## Installation
"""
logger.info("# Object Detection")

pip install -U mlx-vlm

"""
## Import Dependencies
"""
logger.info("## Import Dependencies")


"""
## Qwen2-VL
"""
logger.info("## Qwen2-VL")

model, processor = load("mlx-community/Qwen2-VL-7B-Instruct-8bit")
config = model.config

image = "images/desktop_setup.png"
image = load_image(image)
image

messages = [
    {"role": "system", "content": """
    You are a helpfull assistant to detect objects in images.
     When asked to detect elements based on a description you return bounding boxes for all elements in the form of [xmin, ymin, xmax, ymax].
     When there are more than one result, answer with a list of bounding boxes in the form of [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...].
     Response format:
     ```
     [{
        "object": "object_name",
        "bboxes": [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]
     }, ...]
     ```
    """},
    {"role": "user", "content": "detect all objects in the image"}
]
prompt = apply_chat_template(processor, config, messages)

output = generate(
    model,
    processor,
    prompt,
    image,
    max_tokens=1000,
    temperature=0.7,
    verbose=True,
)

logger.debug(output)

"""
### Plot bounding boxes
"""
logger.info("### Plot bounding boxes")

objects_data = parse_bbox(output, model_type="qwen")
plot_image_with_bboxes(image, bboxes=objects_data, model_type="qwen")

"""
## Paligemma
"""
logger.info("## Paligemma")

model, processor = load("mlx-community/paligemma-3b-mix-224-8bit")
config = model.config

image = "images/cats.jpg"
image = load_image(image)
image

prompt = "detect cat"
prompt = apply_chat_template(processor, config, prompt)

output = generate(
    model,
    processor,
    prompt,
    image,
    verbose=True,
)

output

"""
### Plot bounding boxes
"""
logger.info("### Plot bounding boxes")

bboxes = parse_bbox(output, model_type="paligemma")
bboxes

plot_image_with_bboxes(image, bboxes, model_type="paligemma")

logger.info("\n\n[DONE]", bright=True)