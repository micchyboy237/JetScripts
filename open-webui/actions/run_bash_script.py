from pydantic import BaseModel, Field
from typing import Optional
from fastapi.requests import Request
from subprocess import Popen, PIPE

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

            # Execute Bash code if the input is detected as code
            input_text = last_assistant_message["content"]
            if input_text.startswith("```bash") and input_text.endswith("```"):
                # Remove the ```bash and ``` markers
                code = input_text[7:-3].strip()
                result = self.execute_bash_code(code)
                return {
                    "type": "code_execution_result",
                    "data": {
                        "is_success": result["is_success"],
                        "returncode": result["returncode"],
                        "output": result["data"],
                        "done": True
                    }
                }

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
