from jet.logger import CustomLogger
import GalleryPage from '../src/components/GalleryPage';
import os
import shutil
import {findAllNotebooks} from '../src/components/NotebookUtils';


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
hide_table_of_contents: true
---


# Notebooks

This page contains a collection of notebooks that demonstrate how to use
AutoGen. The notebooks are tagged with the topics they cover.
For example, a notebook that demonstrates how to use function calling will
be tagged with `tool/function`.

<GalleryPage items={findAllNotebooks()} target="_self" allowDefaultImage={false}/>
"""
logger.info("# Notebooks")

logger.info("\n\n[DONE]", bright=True)