from PIL import Image
from PIL import ImageDraw
from jet.logger import CustomLogger
from mlx_vlm import load, generate
import mlx.core as mx
import numpy as np
import os
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# OCR with Region


In this example, you will learn how to perform OCR with Region using the MLX-VLM (Vision Language Model) library. 

We will use `Microsoft's Florence-2`, for OCR with Region in images.

## Installation
"""
logger.info("# OCR with Region")

pip install -U mlx-vlm

"""
## Import Dependencies
"""
logger.info("## Import Dependencies")


"""
## Florence-2
"""
logger.info("## Florence-2")

model_id = 'mlx-community/Florence-2-base-ft-8bit'
model, processor = load(model_id, {"trust_remote_code": True})

"""
## OCR with Region
"""
logger.info("## OCR with Region")

prompt = "<OCR_WITH_REGION>"
image = "images/menu.webp"

generated_text = generate(model, processor, image, prompt, temp=0.8, max_tokens=100000, verbose=True)
image = Image.open(image)
parsed_answer = processor.post_process_generation("".join(generated_text) + "</s>", task=prompt, image_size=(image.width, image.height))


colormap = ['red', 'blue', 'green', 'gray', 'purple', 'orange', 'pink', 'brown', 'gray', 'black']
def draw_ocr_bboxes(image, prediction, scale=1):
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0]+8, new_box[1]+2),
                    "{}".format(label),
                    align="right",

                    fill=color)

    display(image)


draw_ocr_bboxes(image, parsed_answer["<OCR_WITH_REGION>"])

logger.info("\n\n[DONE]", bright=True)