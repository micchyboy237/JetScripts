from IPython.core.display import Image
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.vertex import VertexMultiModalEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Vertex AI Multimodal Embedding
Uses APPLICATION_DEFAULT_CREDENTIALS if no credentials is specified.
"""
logger.info("# Vertex AI Multimodal Embedding")


embed_model = VertexMultiModalEmbedding(
    project="speedy-atom-413006", location="us-central1"
)

image_url = "https://upload.wikimedia.org/wikipedia/commons/4/43/Cute_dog.jpg"

"""
Download this image to `data/test-image.jpg`
"""
logger.info("Download this image to `data/test-image.jpg`")


display(Image(url=image_url, width=500))

result = embed_model.get_image_embedding(f"{GENERATED_DIR}/test-image.jpg")

result[:10]

text_result = embed_model.get_text_embedding(
    "a brown and white puppy laying in the grass with purple daisies in the background"
)

text_result_2 = embed_model.get_text_embedding("airplanes in the sky")

"""
We expect that a similar description to the image will yield a higher similarity result
"""
logger.info("We expect that a similar description to the image will yield a higher similarity result")

embed_model.similarity(result, text_result)

embed_model.similarity(result, text_result_2)

logger.info("\n\n[DONE]", bright=True)