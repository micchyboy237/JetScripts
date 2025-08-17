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
title: Overview
icon: "info"
iconType: "solid"
---

Learn about the key features and capabilities that make Mem0 a powerful platform for memory management and retrieval.

## Core Features

<CardGroup>
  <Card title="Advanced Retrieval" icon="magnifying-glass" href="advanced-retrieval">
    Superior search results using state-of-the-art algorithms, including keyword search, reranking, and filtering capabilities.
  </Card>
  <Card title="Contextual Add" icon="square-plus" href="contextual-add">
    Only send your latest conversation history - we automatically retrieve the rest and generate properly contextualized memories.
  </Card>
  <Card title="Multimodal Support" icon="photo-film" href="multimodal-support">
    Process and analyze various types of content including images.
  </Card>
  <Card title="Memory Customization" icon="filter" href="selective-memory">
    Customize and curate stored memories to focus on relevant information while excluding unnecessary data, enabling improved accuracy, privacy control, and resource efficiency.
  </Card>
  <Card title="Custom Categories" icon="tags" href="custom-categories">
    Create and manage custom categories to organize memories based on your specific needs and requirements.
  </Card>
  <Card title="Custom Instructions" icon="list-check" href="custom-instructions">
    Define specific guidelines for your project to ensure consistent handling of information and requirements.
  </Card>
  <Card title="Direct Import" icon="message-bot" href="direct-import">
    Tailor the behavior of your Mem0 instance with custom prompts for specific use cases or domains.
  </Card>
  <Card title="Async Client" icon="bolt" href="async-client">
    Asynchronous client for non-blocking operations and high concurrency applications.
  </Card>
  <Card title="Memory Export" icon="file-export" href="memory-export">
    Export memories in structured formats using customizable Pydantic schemas.
  </Card>
  <Card title="Graph Memory" icon="graph" href="graph-memory">
    Add memories in the form of nodes and edges in a graph database and search for related memories.
  </Card>
</CardGroup>

## Getting Help

If you have any questions about these features or need assistance, our team is here to help:

<Snippet file="get-help.mdx" />
"""
logger.info("## Core Features")

logger.info("\n\n[DONE]", bright=True)