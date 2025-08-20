import asyncio
from jet.transformers.formatters import format_json
from box_sdk_gen import DeveloperTokenConfig, BoxDeveloperTokenAuth, BoxClient
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.box import BoxSearchToolSpec
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



BOX_DEV_TOKEN = "your_box_dev_token"

config = DeveloperTokenConfig(BOX_DEV_TOKEN)
auth = BoxDeveloperTokenAuth(config)
box_client = BoxClient(auth)

# os.environ["OPENAI_API_KEY"] = "your-key"



box_tool_spec = BoxSearchToolSpec(box_client)

agent = FunctionAgent(
    tools=box_tool_spec.to_tool_list(),
    llm=MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)

async def run_async_code_85b19155():
    async def run_async_code_e141acb6():
        answer = await agent.run("search all invoices")
        return answer
    answer = asyncio.run(run_async_code_e141acb6())
    logger.success(format_json(answer))
    return answer
answer = asyncio.run(run_async_code_85b19155())
logger.success(format_json(answer))
logger.debug(answer)

"""
```
I found the following invoices:

1. **Invoice-A5555.txt**
   - Size: 150 bytes
   - Created By: RB Admin
   - Created At: 2024-04-30 06:22:18

2. **Invoice-Q2468.txt**
   - Size: 176 bytes
   - Created By: RB Admin
   - Created At: 2024-04-30 06:22:19

3. **Invoice-B1234.txt**
   - Size: 168 bytes
   - Created By: RB Admin
   - Created At: 2024-04-30 06:22:15

4. **Invoice-Q8888.txt**
   - Size: 164 bytes
   - Created By: RB Admin
   - Created At: 2024-04-30 06:22:14

5. **Invoice-C9876.txt**
   - Size: 189 bytes
   - Created By: RB Admin
   - Created At: 2024-04-30 06:22:17

These are the invoices found in the search. Let me know if you need more information or assistance with these invoices.
```
"""
logger.info("I found the following invoices:")

logger.info("\n\n[DONE]", bright=True)