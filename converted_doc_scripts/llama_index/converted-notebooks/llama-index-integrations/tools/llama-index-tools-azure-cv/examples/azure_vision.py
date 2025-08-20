from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.azure_cv.base import AzureCVToolSpec
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


# os.environ["OPENAI_API_KEY"] = "sk-proj-1234567890"



cv_tool = AzureCVToolSpec(api_key="your-key", resource="your-resource")

agent = FunctionAgent(
    tools=cv_tool.to_tool_list(),
    llm=MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
)

logger.debug(
    await agent.run(
        "caption this image and tell me what tags are in it"
        " https://portal.vision.cognitive.azure.com/dist/assets/ImageCaptioningSample1-bbe41ac5.png"
    )
)

logger.info("\n\n[DONE]", bright=True)