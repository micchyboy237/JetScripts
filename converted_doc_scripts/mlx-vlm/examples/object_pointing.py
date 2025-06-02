from jet.logger import CustomLogger
from mlx_vlm import load, apply_chat_template, generate
from mlx_vlm.utils import load_image
from utils import parse_points, plot_locations
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Object Pointing and Counting


In this example, you will learn how to perform object pointing and counting using the MLX-VLM (Vision Language Model) library. 

We will use `Molmo-7B-D-0924`, for pointing and counting objects in images.

## Installation
"""
logger.info("# Object Pointing and Counting")

pip install -U mlx-vlm

"""
## Import Dependencies
"""
logger.info("## Import Dependencies")


"""
## Molmo-7B-D-0924
"""
logger.info("## Molmo-7B-D-0924")

model, processor = load("mlx-community/Molmo-7B-D-0924-4bit")
config = model.config

image = "images/desktop_setup.png"
image = load_image(image)
image

"""
## Point to all objects
"""
logger.info("## Point to all objects")

messages = [
    {"role": "user", "content": "Point out to all speakers"}
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
### Count objects
"""
logger.info("### Count objects")

x, y, item_labels = parse_points(output)

logger.debug(f"Number of objects: {len(x)}")

"""
### Plot locations
"""
logger.info("### Plot locations")

plot_locations(output, image)

"""
## Point to specific objects
"""
logger.info("## Point to specific objects")

messages = [
    {"role": "user", "content": "Point out to left speaker, keyboard, monitor, mouse, right speaker"}
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
### Counting objects
"""
logger.info("### Counting objects")

x, y, item_labels = parse_points(output)

logger.debug(f"Number of objects: {len(x)}")

"""
### Plot locations
"""
logger.info("### Plot locations")

plot_locations(output, image)

logger.info("\n\n[DONE]", bright=True)