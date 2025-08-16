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
# Azure Cosmos DB

> "MLX relies on Cosmos DB to dynamically scale their ChatGPT service – one of the fastest-growing consumer apps ever – enabling high reliability and low maintenance."
> – Satya Nadella, Microsoft chairman and chief executive officer

Azure Cosmos DB is a fully managed [NoSQL](https://learn.microsoft.com/en-us/azure/cosmos-db/distributed-nosql), [relational](https://learn.microsoft.com/en-us/azure/cosmos-db/distributed-relational), and [vector database](https://learn.microsoft.com/azure/cosmos-db/vector-database). It offers single-digit millisecond response times, automatic and instant scalability, along with guaranteed speed at any scale. Your business continuity is assured with up to 99.999% availability backed by SLA.

Your can simplify your application development by using this single database service for all your AI agent memory system needs, from [geo-replicated distributed cache](https://medium.com/@marcodesanctis2/using-azure-cosmos-db-as-your-persistent-geo-replicated-distributed-cache-b381ad80f8a0) to tracing/logging to [vector database](https://learn.microsoft.com/en-us/azure/cosmos-db/vector-database).

Learn more about how Azure Cosmos DB enhances the performance of your [AI agent](https://learn.microsoft.com/en-us/azure/cosmos-db/ai-agents).

- [Try Azure Cosmos DB free](https://learn.microsoft.com/en-us/azure/cosmos-db/try-free)
- [Use Azure Cosmos DB lifetime free tier](https://learn.microsoft.com/en-us/azure/cosmos-db/free-tier)
"""
logger.info("# Azure Cosmos DB")

logger.info("\n\n[DONE]", bright=True)