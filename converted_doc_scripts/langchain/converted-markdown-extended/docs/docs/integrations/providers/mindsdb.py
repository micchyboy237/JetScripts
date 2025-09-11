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
# MindsDB

MindsDB is the platform for customizing AI from enterprise data. With MindsDB and it's nearly 200 integrations to [data sources](https://docs.mindsdb.com/integrations/data-overview) and [AI/ML frameworks](https://docs.mindsdb.com/integrations/ai-overview), any developer can use their enterprise data to customize AI for their purpose, faster and more securely.

With MindsDB, you can connect any data source to any AI/ML model to implement and automate AI-powered applications. Deploy, serve, and fine-tune models in real-time, utilizing data from databases, vector stores, or applications. Do all that using universal tools developers already know.

MindsDB integrates with LangChain, enabling users to:


- Deploy models available via LangChain within MindsDB, making them accessible to numerous data sources.
- Fine-tune models available via LangChain within MindsDB using real-time and dynamic data.
- Automate AI workflows with LangChain and MindsDB.

Follow [our docs](https://docs.mindsdb.com/integrations/ai-engines/langchain) to learn more about MindsDB’s integration with LangChain and see examples.
"""
logger.info("# MindsDB")

logger.info("\n\n[DONE]", bright=True)