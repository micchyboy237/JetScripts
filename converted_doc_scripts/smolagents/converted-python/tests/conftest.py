from jet.logger import logger
from smolagents.agents import MultiStepAgent
from smolagents.monitoring import LogLevel
from unittest.mock import patch
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





# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.agents", "tests.fixtures.tools"]

original_multi_step_agent_init = MultiStepAgent.__init__


@pytest.fixture(autouse=True)
def patch_multi_step_agent_with_suppressed_logging():
    with patch.object(MultiStepAgent, "__init__", autospec=True) as mock_init:

        def init_with_suppressed_logging(self, *args, verbosity_level=LogLevel.OFF, **kwargs):
            original_multi_step_agent_init(self, *args, verbosity_level=verbosity_level, **kwargs)

        mock_init.side_effect = init_with_suppressed_logging
        yield

logger.info("\n\n[DONE]", bright=True)