from importlib.util import find_spec
from jet.logger import logger
import os
import pytest
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

"""Markers for tests ."""
require_run_all = pytest.mark.skipif(not os.getenv("RUN_ALL"), reason="requires RUN_ALL environment variable")
require_soundfile = pytest.mark.skipif(find_spec("soundfile") is None, reason="requires soundfile")
require_torch = pytest.mark.skipif(find_spec("torch") is None, reason="requires torch")

logger.info("\n\n[DONE]", bright=True)