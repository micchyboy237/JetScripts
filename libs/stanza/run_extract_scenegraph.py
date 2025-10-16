"""
Very short demo for the SceneGraph interface in the CoreNLP server

Requires CoreNLP >= 4.5.5, Stanza >= 1.5.1
"""


from jet.file.utils import save_file
from jet.transformers.formatters import format_json
from stanza.server import CoreNLPClient

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

if __name__ == "__main__":
    # start_server=None if you have the server running in another process on the same host
    # you can start it with whatever normal options CoreNLPClient has
    #
    # preload=False avoids having the server unnecessarily load annotators
    # if you don't plan on using them
    with CoreNLPClient(preload=False) as client:
        text = "Jennifer's antennae are on her head."
        scenegraph_result = client.scenegraph(text)
        logger.debug(f"Text:\n{text}")
        logger.success(format_json(scenegraph_result))
        save_file(scenegraph_result, f"{OUTPUT_DIR}/scenegraph_result.json")


