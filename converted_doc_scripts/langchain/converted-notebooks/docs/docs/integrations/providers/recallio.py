from jet.logger import logger
from langchain_recallio.memory import RecallioMemory
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
# Recallio

[Recallio](https://recallio.ai/) is a powerfull API allowing to store, index, and retrieve application “memories” with built-in fact extraction, dynamic summaries, reranked recall, and a full knowledge-graph layer.


## Installation

```bash
pip install langchain-recallio
```

```python
```
"""
logger.info("# Recallio")

logger.info("\n\n[DONE]", bright=True)