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

>[Blackboard Learn](https://en.wikipedia.org/wiki/Blackboard_Learn) (previously the Blackboard Learning Management System) is a web-based virtual learning environment and learning management system developed by Blackboard Inc. The software features course management, customizable open architecture, and scalable design that allows integration with student information systems and authentication protocols. It may be installed on local servers, hosted by `Blackboard ASP Solutions`, or provided as Software as a Service hosted on Amazon Web Services. Its main purposes are stated to include the addition of online elements to courses traditionally delivered face-to-face and development of completely online courses with few or no face-to-face meetings

This covers how to load data from a [Blackboard Learn](https://www.anthology.com/products/teaching-and-learning/learning-effectiveness/blackboard-learn) instance.

This loader is not compatible with all `Blackboard` courses. It is only
    compatible with courses that use the new `Blackboard` interface.
    To use this loader, you must have the BbRouter cookie. You can get this
    cookie by logging into the course and then copying the value of the
    BbRouter cookie from the browser's developer tools.
"""
logger.info("# Blackboard")


loader = BlackboardLoader(
    blackboard_course_url="https://blackboard.example.com/webapps/blackboard/execute/announcement?method=search&context=course_entry&course_id=_123456_1",
    bbrouter="expires:12345...",
    load_all_recursively=True,
)
documents = loader.load()

logger.info("\n\n[DONE]", bright=True)