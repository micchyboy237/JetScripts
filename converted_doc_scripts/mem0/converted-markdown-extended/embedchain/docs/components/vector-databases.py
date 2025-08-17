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
title: 🗄️ Vector databases
---

## Overview

Utilizing a vector database alongside Embedchain is a seamless process. All you need to do is configure it within the YAML configuration file. We've provided examples for each supported database below:

<CardGroup cols={4}>
  <Card title="ChromaDB" href="#chromadb"></Card>
  <Card title="Elasticsearch" href="#elasticsearch"></Card>
  <Card title="OpenSearch" href="#opensearch"></Card>
  <Card title="Zilliz" href="#zilliz"></Card>
  <Card title="LanceDB" href="#lancedb"></Card>
  <Card title="Pinecone" href="#pinecone"></Card>
  <Card title="Qdrant" href="#qdrant"></Card>
  <Card title="Weaviate" href="#weaviate"></Card>
</CardGroup>

<Snippet file="missing-vector-db-tip.mdx" />
"""
logger.info("## Overview")

logger.info("\n\n[DONE]", bright=True)