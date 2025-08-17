from embedchain import App
from embedchain import Pipeline as App
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: LanceDB
---

## Install Embedchain with LanceDB

Install Embedchain, LanceDB and  related dependencies using the following command:
"""
logger.info("## Install Embedchain with LanceDB")

pip install "embedchain[lancedb]"

"""
LanceDB is a developer-friendly, open source database for AI. From hyper scalable vector search and advanced retrieval for RAG, to streaming training data and interactive exploration of large scale AI datasets.
In order to use LanceDB as vector database, not need to set any key for local use. 

### With OPENAI 
<CodeGroup>
"""
logger.info("### With OPENAI")


# os.environ["OPENAI_API_KEY"] = "sk-xxx"

app = App.from_config(config={
    "vectordb": {
        "provider": "lancedb",
            "config": {
                "collection_name": "lancedb-index"
            }
        }
    }
)

app.add("https://www.forbes.com/profile/elon-musk")

while(True):
    question = input("Enter question: ")
    if question in ['q', 'exit', 'quit']:
        break
    answer = app.query(question)
    logger.debug(answer)

"""
</CodeGroup>

### With Local LLM 
<CodeGroup>
"""
logger.info("### With Local LLM")


config = {
  'llm': {
    'provider': 'huggingface',
    'config': {
      'model': 'mistralai/Mistral-7B-v0.1',
      'temperature': 0.1,
      'max_tokens': 250,
      'top_p': 0.1,
      'stream': True
    }
  },
  'embedder': {
    'provider': 'huggingface',
    'config': {
      'model': 'sentence-transformers/all-mpnet-base-v2'
    }
  },
  'vectordb': {
    'provider': 'lancedb',
    'config': {
      'collection_name': 'lancedb-index'
    }
  }
}

app = App.from_config(config=config)

app.add("https://www.tesla.com/ns_videos/2022-tesla-impact-report.pdf")

while(True):
    question = input("Enter question: ")
    if question in ['q', 'exit', 'quit']:
        break
    answer = app.query(question)
    logger.debug(answer)

"""
</CodeGroup>


<Snippet file="missing-vector-db-tip.mdx" />
"""

logger.info("\n\n[DONE]", bright=True)