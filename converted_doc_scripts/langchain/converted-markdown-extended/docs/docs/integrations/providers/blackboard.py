from jet.logger import logger
from langchain_community.document_loaders import BlackboardLoader
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
# Blackboard

>[Blackboard Learn](https://en.wikipedia.org/wiki/Blackboard_Learn) (previously the `Blackboard Learning Management System`)
> is a web-based virtual learning environment and learning management system developed by Blackboard Inc.
> The software features course management, customizable open architecture, and scalable design that allows
> integration with student information systems and authentication protocols. It may be installed on local servers,
> hosted by `Blackboard ASP Solutions`, or provided as Software as a Service hosted on Amazon Web Services.
> Its main purposes are stated to include the addition of online elements to courses traditionally delivered
> face-to-face and development of completely online courses with few or no face-to-face meetings.

## Installation and Setup

There isn't any special setup for it.

## Document Loader

See a [usage example](/docs/integrations/document_loaders/blackboard).
"""
logger.info("# Blackboard")


logger.info("\n\n[DONE]", bright=True)