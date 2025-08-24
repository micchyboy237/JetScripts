import os
from mlx_vlm import load, apply_chat_template, generate
from mlx_vlm.utils import load_image
from mlx_vlm.utils import process_image
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)

if __name__ == "__main__":
    images = [
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/mlx-vlm/examples/images/cats.jpg",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/mlx-vlm/examples/images/desktop_setup.png",
    ]

    messages = [
        {"role": "user", "content": "Describe what you see in the images."}
    ]

    # Load model and processor
    qwen_vl_model, qwen_vl_processor = load(
        "mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
    # qwen_vl_model, qwen_vl_processor = load("mlx-community/Qwen2.5-VL-3B-Instruct-3bit")
    qwen_vl_config = qwen_vl_model.config

    prompt = apply_chat_template(
        qwen_vl_processor, qwen_vl_config, messages, num_images=len(images))

    qwen_vl_output = generate(
        qwen_vl_model,
        qwen_vl_processor,
        prompt,
        images,
        max_tokens=1000,
        temperature=0.7,
        verbose=True
    )

    save_file(qwen_vl_output, f"{OUTPUT_DIR}/qwen_vl_output.json")
