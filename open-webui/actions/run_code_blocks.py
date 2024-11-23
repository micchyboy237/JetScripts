import os
import tempfile
import venv
from pydantic import BaseModel, Field
from typing import Optional
from fastapi.requests import Request
from subprocess import Popen, PIPE
from autogen.code_utils import create_virtual_env
from autogen.coding import CodeBlock, MarkdownCodeExtractor, LocalCommandLineCodeExecutor

"""
MODERATOR COMMENT: This function should be utilized with EXTREME caution.
Do not expose to untrusted users or deploy on secure networks unless you are sure you have considered all risks.
"""


class Action:
    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        show_status: bool = Field(
            default=True, description="Show status of the action."
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    def execute_bash_code(self, code: str) -> dict:
        """Executes Bash code and returns the result."""
        try:
            process = Popen(["bash", "-c", code], stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                return {"is_success": False, "returncode": process.returncode, "error": stderr.decode('utf-8')}
            return {"is_success": True, "returncode": 0, "data": stdout.decode("utf-8")}
        except Exception as e:
            return {"is_success": False, "returncode": -1, "error": str(e)}

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:
        print(f"action:{__name__}")

        user_valves = __user__.get("valves")
        if not user_valves:
            user_valves = self.UserValves()

        if __event_emitter__:
            last_assistant_message = body["messages"][-1]

            if user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Processing your input", "done": False},
                    }
                )

            # Find all code blocks from last AI response
            input_text = last_assistant_message["content"]
            extractor = MarkdownCodeExtractor()
            code_blocks = extractor.extract_code_blocks(input_text)

            with tempfile.TemporaryDirectory() as temp_dir:
                env_builder = venv.EnvBuilder(with_pip=True)
                env_builder.create(temp_dir)
                env_builder_context = env_builder.ensure_directories(temp_dir)

                executor = LocalCommandLineCodeExecutor(
                    work_dir=temp_dir, virtual_env_context=env_builder_context)
                execution = executor.execute_code_blocks(code_blocks)

                assert execution.exit_code == 0
                assert execution.output.strip() == "True"

            if user_valves.show_status:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "No valid Bash code detected",
                            "done": True,
                        },
                    }
                )
