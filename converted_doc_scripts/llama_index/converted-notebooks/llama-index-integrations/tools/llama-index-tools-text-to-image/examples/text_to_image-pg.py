from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.core.workflow import Context
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms import MLX
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.tools.text_to_image.base import TextToImageToolSpec
import os
import requests
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



# os.environ["OPENAI_API_KEY"] = "sk-..."



response = requests.get(
    "https://www.dropbox.com/s/f6bmb19xdg0xedm/paul_graham_essay.txt?dl=1"
)
essay_txt = response.text
with open("pg_essay.txt", "w") as fp:
    fp.write(essay_txt)

documents = SimpleDirectoryReader(input_files=["pg_essay.txt"]).load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="paul_graham",
        description=(
            "Provides a biography of Paul Graham, from childhood to college to adult"
            " life"
        ),
    ),
)


llm = MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

text_to_image_spec = TextToImageToolSpec()
tools = text_to_image_spec.to_tool_list()
agent = FunctionAgent(
    tools=tools + [query_engine_tool], llm=llm
)

ctx = Context(agent)

logger.debug(
    await agent.run(
        "generate an image of the car that Paul Graham bought after Yahoo bought his"
        " company",
        ctx=ctx
    )
)

logger.info("\n\n[DONE]", bright=True)