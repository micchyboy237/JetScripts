from jet.logger import logger
from langchain_writer import ChatWriter
from langchain_writer.pdf_parser import PDFParser
from langchain_writer.text_splitter import WriterTextSplitter
from langchain_writer.tools import GraphTool
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
---
keywords: [writer]
---

# Writer, Inc.

All functionality related to Writer


>This page covers how to use the [Writer](https://writer.com/) ecosystem within LangChain. For further information see Writer [docs](https://dev.writer.com/home/introduction).
>[Palmyra](https://writer.com/blog/palmyra/) is a Large Language Model (LLM) developed by `Writer, Inc`.
>
>The [Writer API](https://dev.writer.com/api-guides/introduction) is powered by a diverse set of Palmyra sub-models with different capabilities and price points.

## Installation and Setup

Install the integration package with
"""
logger.info("# Writer, Inc.")

pip install langchain-writer

"""
Get an Writer API key and set it as an environment variable (`WRITER_API_KEY`)

## Chat model
"""
logger.info("## Chat model")


"""
See [details](/docs/integrations/chat/writer).

## PDF Parser
"""
logger.info("## PDF Parser")


"""
See [details](/docs/integrations/document_loaders/parsers/writer_pdf_parser).

## Text splitter
"""
logger.info("## Text splitter")


"""
See [details](/docs/integrations/splitters/writer_text_splitter).

## Tools calling

### Functions

Support of basic function calls defined via dicts, Pydantic, python functions etc.

### Graphs
"""
logger.info("## Tools calling")


"""
See [details](/docs/integrations/tools/writer).

Writer-specific remotely invoking tool
"""
logger.info("See [details](/docs/integrations/tools/writer).")

logger.info("\n\n[DONE]", bright=True)