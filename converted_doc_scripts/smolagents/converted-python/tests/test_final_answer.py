from .test_tools import ToolTesterMixin
from .utils.markers import require_torch
from jet.logger import logger
from smolagents.agent_types import _AGENT_TYPE_MAPPING
from smolagents.default_tools import FinalAnswerTool
import PIL.Image
import numpy as np
import os
import pytest
import shutil
import torch


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.






class TestFinalAnswerTool(ToolTesterMixin):
    def setup_method(self):
        self.inputs = {"answer": "Final answer"}
        self.tool = FinalAnswerTool()

    def test_exact_match_arg(self):
        result = self.tool("Final answer")
        assert result == "Final answer"

    def test_exact_match_kwarg(self):
        result = self.tool(answer=self.inputs["answer"])
        assert result == "Final answer"

    @require_torch
    def test_agent_type_output(self, inputs):
        for input_type, input in inputs.items():
            output = self.tool(**input, sanitize_inputs_outputs=True)
            agent_type = _AGENT_TYPE_MAPPING[input_type]
            assert isinstance(output, agent_type)

    @pytest.fixture
    def inputs(self, shared_datadir):

        return {
            "string": {"answer": "Text input"},
            "image": {"answer": PIL.Image.open(shared_datadir / "000000039769.png").resize((512, 512))},
            "audio": {"answer": torch.Tensor(np.ones(3000))},
        }

logger.info("\n\n[DONE]", bright=True)