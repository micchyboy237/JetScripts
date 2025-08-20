from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.dashscope import (
DashScopeEmbedding,
DashScopeBatchTextEmbeddingModels,
DashScopeTextEmbeddingType,
)
from llama_index.embeddings.dashscope import (
DashScopeEmbedding,
DashScopeMultiModalEmbeddingModels,
)
from llama_index.embeddings.dashscope import (
DashScopeEmbedding,
DashScopeTextEmbeddingModels,
DashScopeTextEmbeddingType,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/dashscope_embeddings.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""



"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.")

# %pip install llama-index-core
# %pip install llama-index-embeddings-dashscope

# %env DASHSCOPE_API_KEY=YOUR_DASHSCOPE_API_KEY


embedder = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
)
text_to_embedding = ["风急天高猿啸哀", "渚清沙白鸟飞回", "无边落木萧萧下", "不尽长江滚滚来"]
result_embeddings = embedder.get_text_embedding_batch(text_to_embedding)
for index, embedding in enumerate(result_embeddings):
    if embedding is None:  # if the correspondence request is embedding failed.
        logger.debug("The %s embedding failed." % text_to_embedding[index])
    else:
        logger.debug("Dimension of embeddings: %s" % len(embedding))
        logger.debug(
            "Input: %s, embedding is: %s"
#             % (text_to_embedding[index], embedding[:5])
        )


embedder = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_QUERY,
)
embedding = embedder.get_text_embedding("衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买")
logger.debug(f"Dimension of embeddings: {len(embedding)}")
logger.debug(embedding[:5])


embedder = DashScopeEmbedding(
    model_name=DashScopeBatchTextEmbeddingModels.TEXT_EMBEDDING_ASYNC_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
)

embedding_result_file_url = embedder.get_batch_text_embedding(
    embedding_file_url="https://dashscope.oss-cn-beijing.aliyuncs.com/samples/text/text-embedding-test.txt"
)
logger.debug(embedding_result_file_url)


embedder = DashScopeEmbedding(
    model_name=DashScopeMultiModalEmbeddingModels.MULTIMODAL_EMBEDDING_ONE_PEACE_V1,
)

embedding = embedder.get_image_embedding(
    img_file_path="https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png"
)
logger.debug(f"Dimension of embeddings: {len(embedding)}")
logger.debug(embedding[:5])


embedder = DashScopeEmbedding(
    model_name=DashScopeMultiModalEmbeddingModels.MULTIMODAL_EMBEDDING_ONE_PEACE_V1,
)

input = [
    {"factor": 1, "text": "你好"},
    {
        "factor": 2,
        "audio": "https://dashscope.oss-cn-beijing.aliyuncs.com/audios/cow.flac",
    },
    {
        "factor": 3,
        "image": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png",
    },
]

embedding = embedder.get_multimodal_embedding(input=input)
logger.debug(f"Dimension of embeddings: {len(embedding)}")
logger.debug(embedding[:5])

logger.info("\n\n[DONE]", bright=True)