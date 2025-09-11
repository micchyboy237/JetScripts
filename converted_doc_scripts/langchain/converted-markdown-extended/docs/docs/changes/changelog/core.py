from jet.logger import logger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# langchain-core

## 0.1.x

#### Deprecated

- `BaseChatModel` methods `__call__`, `call_as_llm`, `predict`, `predict_messages`. Will be removed in 0.2.0. Use `BaseChatModel.invoke` instead.
- `BaseChatModel` methods `apredict`, `apredict_messages`. Will be removed in 0.2.0. Use `BaseChatModel.ainvoke` instead.
- `BaseLLM` methods `__call__`, `predict`, `predict_messages`. Will be removed in 0.2.0. Use `BaseLLM.invoke` instead.
- `BaseLLM` methods `apredict`, `apredict_messages`. Will be removed in 0.2.0. Use `BaseLLM.ainvoke` instead.
"""
logger.info("# langchain-core")

logger.info("\n\n[DONE]", bright=True)