from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.mixedbreadai import MixedbreadAIEmbedding
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/mixedbreadai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Mixedbread AI Embeddings

Explore the capabilities of MixedBread AI's embedding models with custom encoding formats (binary, int, float, base64, etc.), embedding dimensions (Matryoshka) and context prompts.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Mixedbread AI Embeddings")

# %pip install llama-index-embeddings-mixedbreadai

# !pip install llama-index


mixedbread_api_key = os.environ.get("MXBAI_API_KEY", "your-api-key")

model_name = "mixedbread-ai/mxbai-embed-large-v1"

oven = MixedbreadAIEmbedding(api_key=mixedbread_api_key, model_name=model_name)

embeddings = oven.get_query_embedding("Why bread is so tasty?")

logger.debug(len(embeddings))
logger.debug(embeddings[:5])

"""
### Using prompt for contextual embedding

The prompt can improve the model's understanding of how the embedding will be used in subsequent tasks, which in turn increases the performance. Our experiments show that having domain specific prompts can increase the performance.
"""
logger.info("### Using prompt for contextual embedding")

prompt_for_retrieval = (
    "Represent this sentence for searching relevant passages:"
)

contextual_oven = MixedbreadAIEmbedding(
    api_key=mixedbread_api_key,
    model_name=model_name,
    prompt=prompt_for_retrieval,
)

contextual_embeddings = contextual_oven.get_query_embedding(
    "What bread is invented in Germany?"
)

logger.debug(len(contextual_embeddings))
logger.debug(contextual_embeddings[:5])

"""
## Quantization and Matryoshka support

The Mixedbread AI embedding supports quantization and matryoshka to reduce the size of embeddings for better storage while retaining most of the performance.
See these posts for more information: 
* [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization)
* [64 bytes per embedding, yee-haw](https://www.mixedbread.ai/blog/binary-mrl).

### Using different encoding formats

The default `encoding_format` is `float`. We also support `float16`, `binary`, `ubinary`, `int8`, `uint8`, `base64`.
"""
logger.info("## Quantization and Matryoshka support")

binary_oven = MixedbreadAIEmbedding(
    api_key=mixedbread_api_key,
    model_name=model_name,
    encoding_format="binary",
)

binary_embeddings = binary_oven.get_text_embedding(
    "The bread is tiny but still filling!"
)

logger.debug(len(binary_embeddings))
logger.debug(binary_embeddings[:5])

"""
### Using different embedding dimensions

Mixedbread AI embedding models support Matryoshka dimension truncation. The default dimension is set to the model's maximum.
Keep an eye on our website to see what models support Matryoshka.
"""
logger.info("### Using different embedding dimensions")

half_oven = MixedbreadAIEmbedding(
    api_key=mixedbread_api_key,
    model_name=model_name,
    dimensions=512,  # 1024 is the maximum of `mxbai-embed-large-v1`
)

half_embeddings = half_oven.get_text_embedding(
    "I want the better half of my bread."
)

logger.debug(len(half_embeddings))
logger.debug(half_embeddings[:5])

logger.info("\n\n[DONE]", bright=True)