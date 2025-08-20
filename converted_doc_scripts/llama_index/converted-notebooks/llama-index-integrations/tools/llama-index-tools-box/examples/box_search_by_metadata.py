import asyncio
from jet.transformers.formatters import format_json
from box_sdk_gen import DeveloperTokenConfig, BoxDeveloperTokenAuth, BoxClient
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.box import (
BoxSearchByMetadataToolSpec,
BoxSearchByMetadataOptions,
)
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




from_ = "enterprise_" + "your_box_enterprise_id" + "." + "your_metadata_template_key"
ancestor_folder_id = "your_starting_folder_id"
query = "documentType = :docType "  # Your metadata query string
query_params = '{"docType": "Invoice"}'  # Your metadata query parameters

options = BoxSearchByMetadataOptions(
    from_=from_,
    ancestor_folder_id=ancestor_folder_id,
    query=query,
)

box_tool = BoxSearchByMetadataToolSpec(box_client=box_client, options=options)

agent = FunctionAgent(
    tools=box_tool.to_tool_list(),
    llm=MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)

async def async_func_38():
    answer = await agent.run(
        f"search all documents using the query_params as the key value pair of  {query_params} "
    )
    return answer
answer = asyncio.run(async_func_38())
logger.success(format_json(answer))
logger.debug(answer)

"""
```
tests/test_tools_box_search_by_metadata.py Added user message to memory: search all documents using the query_params as the key value pair of  {"docType": "Invoice"} 
=== Calling Function ===
Calling function: search with args: {"query_params":"{\"docType\": \"Invoice\"}"}
========================
I found the following documents with the query parameter "docType: Invoice":

1. Document ID: ee053400-e501-49d0-9e9f-e021bd86d9c6
   - Name: Invoice-B1234.txt
   - Size: 168 bytes
   - Created By: RB Admin
   - Created At: 2024-04-30 06:22:15

2. Document ID: 475cfee1-557a-4a3b-bc7c-18634d3a5d99
   - Name: Invoice-C9876.txt
   - Size: 189 bytes
   - Created By: RB Admin
   - Created At: 2024-04-30 06:22:17

3. Document ID: 9dbfd4ec-639f-4c7e-8045-645866e780a3
   - Name: Invoice-Q2468.txt
   - Size: 176 bytes
   - Created By: RB Admin
   - Created At: 2024-04-30 06:22:19

These are the documents matching the query parameter. Let me know if you need more information or assistance.

```
"""
logger.info("tests/test_tools_box_search_by_metadata.py Added user message to memory: search all documents using the query_params as the key value pair of  {"docType": "Invoice"}")

logger.info("\n\n[DONE]", bright=True)