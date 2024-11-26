from pydantic import BaseModel, Field
from typing import Any, Awaitable, Callable, Optional
import aiohttp
import json


class Action:
    class Valves(BaseModel):
        openai_api_url: str = Field(
            default="http://jetairm1:11434",
            description="OpenAI-compatible endpoint",
        )
        model: str = Field(
            default="llama3.1:latest",
            description="Model to use for task generation",
        )
        pass

    class UserValves(BaseModel):
        show_status: bool = Field(
            default=True, description="Show status of the action."
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    def extract_json(self, response: str) -> list:
        """Extract JSON blocks from the response."""
        import re
        matches = re.findall(
            r'```(?:json)?\s*\n(.*?)\n\s*```', response, re.DOTALL)

        if not matches:
            return ["No json found in response."]

        for match in matches:
            json_str = match.strip()
            try:
                json_obj = json.loads(json_str)
                return json_obj
            except json.JSONDecodeError:
                pass

    async def query_openai_api(
        self, model: str, system_prompt: str, user_message: str
    ) -> str:
        """Query the OpenAI API."""
        url = f"{self.valves.openai_api_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "stream": True,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        try:
            async with aiohttp.ClientSession() as session:
                response = await session.post(url, headers=headers, json=payload)
                response.raise_for_status()
                response_data = await response.json()
                return response_data["choices"][0]["message"]["content"]
        except aiohttp.ClientError as e:
            raise Exception(f"HTTP error: {e}")

    async def action(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __event_call__=None,
    ) -> Optional[dict]:
        print(f"action:{__name__}")

        user_valves = __user__.get("valves", self.UserValves())

        last_message = body["messages"][-1]["content"]
        epics = self.extract_json(last_message)

        if __event_emitter__:
            # if user_valves.show_status:
            #     await __event_emitter__(
            #         {
            #             "type": "status",
            #             "data": {"description": f"Generating tasks for epics ({len(epics)})\n{json.dumps(epics)}", "done": False},
            #         }
            #     )

            results = []
            for epic_idx, epic in enumerate(epics):
                if isinstance(epic, dict):
                    epic_title = epic.get("title")
                    epic_description = epic.get("description")

                    if user_valves.show_status:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": f"Generating tasks for epic {epic_idx + 1}", "done": False},
                            }
                        )

                    tasks = []
                    epic_dict = {
                        "title": epic_title,
                        "description": epic_description,
                        "tasks": tasks
                    }
                    results.append(epic_dict)

                    if epic_title and epic_description:
                        system_message = "You are responsible for breaking down epics into actionable tasks. Each task must be specific, include clear acceptance criteria, and define dependencies if any. Format response in JSON array wrapped in ```json."
                        user_prompt = (
                            f"Write tasks for this epic.\nTitle: {epic_title}\nDescription: {epic_description}")
                        task_results_str = await self.query_openai_api(
                            self.valves.model,
                            system_message,
                            user_prompt,
                        )

                        # Check if tasks are valid JSON
                        try:
                            task_results = self.extract_json(task_results_str)
                            tasks.extend(task_results)
                            # Process each task
                            # for task in task_results:
                            #     print(f"Processing task: {task}")
                        except json.JSONDecodeError:
                            print("Invalid task format received.")

            # if user_valves.show_status:
            #     await __event_emitter__(
            #         {
            #             "type": "status",
            #             "data": {
            #                 "description": "Tasks generated and processed successfully."
            #                 if results
            #                 else "Failed to generate tasks.",
            #                 "done": True,
            #             },
            #         }
            #     )
