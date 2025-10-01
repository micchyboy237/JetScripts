"""
Very short demo for the SceneGraph interface in the CoreNLP server

Requires CoreNLP >= 4.5.5, Stanza >= 1.5.1
"""

import json

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

# start_server=None if you have the server running in another process on the same host
# you can start it with whatever normal options CoreNLPClient has
#
# preload=False avoids having the server unnecessarily load annotators
# if you don't plan on using them
with CoreNLPClient(preload=False) as client:
    result = client.scenegraph("Jennifer's antennae are on her head.")
    print(json.dumps(result, indent=2))


