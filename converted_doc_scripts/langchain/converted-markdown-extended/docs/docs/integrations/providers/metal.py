from jet.logger import logger
from langchain.retrievers import MetalRetriever
from metal_sdk.metal import Metal
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
# Metal

This page covers how to use [Metal](https://getmetal.io) within LangChain.

## What is Metal?

Metal is a  managed retrieval & memory platform built for production. Easily index your data into `Metal` and run semantic search and retrieval on it.

![Screenshot of the Metal dashboard showing the Browse Index feature with sample data.](/img/MetalDash.png "Metal Dashboard Interface")

## Quick start

Get started by [creating a Metal account](https://app.getmetal.io/signup).

Then, you can easily take advantage of the `MetalRetriever` class to start retrieving your data for semantic search, prompting context, etc. This class takes a `Metal` instance and a dictionary of parameters to pass to the Metal API.
"""
logger.info("# Metal")



metal = Metal("API_KEY", "CLIENT_ID", "INDEX_ID");
retriever = MetalRetriever(metal, params={"limit": 2})

docs = retriever.invoke("search term")

logger.info("\n\n[DONE]", bright=True)