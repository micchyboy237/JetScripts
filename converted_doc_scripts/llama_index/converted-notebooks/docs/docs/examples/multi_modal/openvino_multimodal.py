from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.multi_modal_llms.openvino import OpenVINOMultiModal
from pathlib import Path
from transformers import AutoProcessor
import gc
import nncf
import openvino as ov
import os
import shutil


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
# Local Multimodal pipeline with OpenVINO

[OpenVINO™](https://github.com/openvinotoolkit/openvino) is an open-source toolkit for optimizing and deploying AI inference. The OpenVINO™ Runtime supports various hardware [devices](https://github.com/openvinotoolkit/openvino?tab=readme-ov-file#supported-hardware-matrix) including x86 and ARM CPUs, and Intel GPUs. It can help to boost deep learning performance in Computer Vision, Automatic Speech Recognition, Natural Language Processing and other common tasks.

Hugging Face multimodal model can be supported by OpenVINO through ``OpenVINOMultiModal`` class.
"""
logger.info("# Local Multimodal pipeline with OpenVINO")

# %pip install llama-index-multi-modal-llms-openvino -q

# %pip install llama-index llama-index-readers-file -q

"""
### Export and compress multimodal model

It is possible to [export your model](https://github.com/huggingface/optimum-intel?tab=readme-ov-file#export) to the OpenVINO IR format with the CLI, and load the model from local folder.
"""
logger.info("### Export and compress multimodal model")


model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
model_path = Path(model_id.split("/")[-1]) / "FP16"

if not model_path.exists():
#     !optimum-cli export openvino --model {model_id} --weight-format fp16 {model_path}


core = ov.Core()

compression_config = {
    "mode": nncf.CompressWeightsMode.INT4_SYM,
    "group_size": 64,
    "ratio": 0.6,
}

compressed_model_path = model_path.parent / "INT4"
if not compressed_model_path.exists():
    ov_model = core.read_model(model_path / "openvino_language_model.xml")
    compressed_ov_model = nncf.compress_weights(ov_model, **compression_config)
    ov.save_model(
        compressed_ov_model,
        compressed_model_path / "openvino_language_model.xml",
    )
    del compressed_ov_model
    del ov_model
    gc.collect()
    for file_name in model_path.glob("*"):
        if file_name.name in [
            "openvino_language_model.xml",
            "openvino_language_model.bin",
        ]:
            continue
        shutil.copy(file_name, compressed_model_path)

"""
### Prepare the input data
"""
logger.info("### Prepare the input data")


os.makedirs("./input_images", exist_ok=True)

url = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

image


processor = AutoProcessor.from_pretrained(
    "llava-v1.6-mistral-7b-hf/INT4", trust_remote_code=True
)


def messages_to_prompt(messages, image_documents):
    """
    Prepares the input messages and images.
    """
    conversation = [{"type": "text", "text": messages[0].content}]
    images = []
    for img_doc in image_documents:
        images.append(img_doc)
        conversation.append({"type": "image"})
    messages = [
        {"role": "user", "content": conversation}
    ]  # Wrap conversation in a user role

    logger.debug(messages)

    text_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True
    )

    inputs = processor(text=text_prompt, images=images, return_tensors="pt")
    return inputs

"""
### Model Loading

Models can be loaded by specifying the model parameters using the `OpenVINOMultiModal` method.

If you have an Intel GPU, you can specify `device_map="gpu"` to run inference on it.
"""
logger.info("### Model Loading")

vlm = OpenVINOMultiModal(
    model_id_or_path="llava-v1.6-mistral-7b-hf/INT4",
    device="cpu",
    messages_to_prompt=messages_to_prompt,
    generate_kwargs={"do_sample": False},
)

"""
### Inference with local OpenVINO model
"""
logger.info("### Inference with local OpenVINO model")

response = vlm.complete("Describe the images", image_documents=[image])
logger.debug(response.text)

"""
### Streaming
"""
logger.info("### Streaming")

response = vlm.stream_complete("Describe the images", image_documents=[image])
for r in response:
    logger.debug(r.delta, end="")

logger.info("\n\n[DONE]", bright=True)