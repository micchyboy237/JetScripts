from ipywidgets import Video
from jet.logger import CustomLogger
from mlx_vlm import load
from mlx_vlm.utils import generate
from mlx_vlm.video_generate import process_vision_info
from pprint import pprint
import mlx.core as mx
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Video Understanding

In this example, we will generate a description of a video using `Qwen2-VL`, `Qwen2-5-VL`, `LLava`, and `Idefics3`, with more models coming soon.

This feature is currently in beta, may not work as expected.

## Install Dependencies
"""
logger.info("# Video Understanding")

# !pip install -U mlx-vlm

"""
## Import Dependencies
"""
logger.info("## Import Dependencies")



model, processor = load("mlx-community/Qwen2.5-VL-7B-Instruct-4bit")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "videos/fastmlx_local_ai_hub.mp4",
                "max_pixels": 360 * 360,
                "fps": 1.0,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

input_ids = mx.array(inputs['input_ids'])
pixel_values = mx.array(inputs['pixel_values_videos'])
mask = mx.array(inputs['attention_mask'])
image_grid_thw = mx.array(inputs['video_grid_thw'])

kwargs = {
    "image_grid_thw": image_grid_thw,
}

kwargs["video"] = "videos/fastmlx_local_ai_hub.mp4"
kwargs["input_ids"] = input_ids
kwargs["pixel_values"] = pixel_values
kwargs["mask"] = mask
response = generate(model, processor, prompt=text, temperature=0.7, max_tokens=100, **kwargs)

plogger.debug(response)

Video.from_file("videos/fastmlx_local_ai_hub.mp4", width=320, height=240)

logger.info("\n\n[DONE]", bright=True)